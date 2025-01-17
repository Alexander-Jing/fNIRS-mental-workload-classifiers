import os
import sys
import numpy as np
import torch
import torch.nn as nn

import argparse
import time

from easydict import EasyDict as edict
from tqdm import trange

YOUR_PATH = os.environ['YOUR_PATH']
sys.path.insert(0, os.path.join(YOUR_PATH, 'fNIRS-mental-workload-classifiers/helpers'))
import models
import brain_data
from utils import generic_GetTrainValTestSubjects, seed_everything, makedir_if_not_exist, plot_confusion_matrix, save_pickle, train_one_epoch, eval_model, save_training_curves_FixedTrainValSplit, write_performance_info_FixedTrainValSplit, write_program_time, write_inference_time
from utils import LabelSmoothing, train_one_epoch_fNIRS_T, eval_model_fNIRST, train_one_epoch_Ours_T, eval_model_OursT
from utils import EarlyStopping, train_one_epoch_Ours_T_chunk, eval_model_OursT_chunk

# from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, help="random seed")
parser.add_argument('--gpu_idx', default=0, type=int, help="gpu idx")
parser.add_argument('--data_dir', default='../data/Leon/Visual/size_40sec_200ts_stride_3ts/', help="folder to the dataset")
parser.add_argument('--window_size', default=200, type=int, help='window size')
parser.add_argument('--result_save_rootdir', default='./experiments', help="Directory containing the dataset")
parser.add_argument('--classification_task', default='four_class', help='binary or four-class classification')
parser.add_argument('--restore_file', default='None', help="xxx.statedict")
parser.add_argument('--n_epoch', default=100, type=int, help="number of epoch")
parser.add_argument('--setting', default='64vs4_TestBucket1', help='which predefined train val test split scenario')


#for personal model, save the test prediction of each cv fold
def train_classifier(args_dict, train_subjects, val_subjects, test_subjects):
    
    #convert to string list
    train_subjects = [str(i) for i in train_subjects]
    val_subjects = [str(i) for i in val_subjects]
    test_subjects = [str(i) for i in test_subjects]
    
    #parse args:
    gpu_idx = args_dict.gpu_idx
    data_dir = args_dict.data_dir
    window_size = args_dict.window_size
    result_save_rootdir = args_dict.result_save_rootdir
    classification_task = args_dict.classification_task
    restore_file = args_dict.restore_file
    n_epoch = args_dict.n_epoch
    
    model_to_use = models.Ours_T_3
    num_chunk_this_window_size = 1488  # the window size 150 has 1488 chunks
    

    if classification_task == 'four_class':
        data_loading_function = brain_data.read_subject_csv
        confusion_matrix_figure_labels = ['0back', '1back', '2back', '3back']
        
    elif classification_task == 'binary':
        data_loading_function = brain_data.read_subject_csv_binary_chunk
        confusion_matrix_figure_labels = ['0back', '2back']
        
    else:
        raise NameError('not supported classification type')
        
    
    #create the group train data 
    group_model_sub_train_feature_list = []
    group_model_sub_train_label_list = []
    
    for subject in train_subjects:
        sub_feature, sub_label = data_loading_function(os.path.join(data_dir, 'sub_{}.csv'.format(subject)),  num_chunk_this_window_size=num_chunk_this_window_size)
        
        group_model_sub_train_feature_list.append(sub_feature)
        group_model_sub_train_label_list.append(sub_label)
    
    group_model_sub_train_feature_array = np.concatenate(group_model_sub_train_feature_list, axis=0).astype(np.float32)
    group_model_sub_train_label_array = np.concatenate(group_model_sub_train_label_list, axis=0)
    
    
    #create the group val data
    group_model_sub_val_feature_list = []
    group_model_sub_val_label_list = []
    
    for subject in val_subjects:
        sub_feature, sub_label = data_loading_function(os.path.join(data_dir, 'sub_{}.csv'.format(subject)),  num_chunk_this_window_size=num_chunk_this_window_size)
        
        group_model_sub_val_feature_list.append(sub_feature)
        group_model_sub_val_label_list.append(sub_label)
    
    group_model_sub_val_feature_array = np.concatenate(group_model_sub_val_feature_list, axis=0).astype(np.float32)
    group_model_sub_val_label_array = np.concatenate(group_model_sub_val_label_list, axis=0)
    
    
    #dataset object
    group_train_set = brain_data.brain_dataset(group_model_sub_train_feature_array, group_model_sub_train_label_array)
    group_val_set = brain_data.brain_dataset(group_model_sub_val_feature_array, group_model_sub_val_label_array)

    #dataloader object
    cv_train_batch_size = len(group_train_set)  # 要看下这个batch size=35712，应该就是整个group_train_set的数据作为一个batch，所以会超显存
    cv_val_batch_size = len(group_val_set)  # 11904
    group_train_loader = torch.utils.data.DataLoader(group_train_set, batch_size=128, shuffle=True) # according to the fNIRS-preT paper, we set the batch size 128
    group_val_loader = torch.utils.data.DataLoader(group_val_set, batch_size=cv_val_batch_size, shuffle=False)
  
    #GPU setting
    cuda = torch.cuda.is_available()
    if cuda:
        print('Detected GPUs', flush = True)
        #device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(gpu_idx))
    else:
        print('DID NOT detect GPUs', flush = True)
        device = torch.device('cpu')
        
    flooding_levels = [[0.55, 0.5, 0.45], [0.5, 0.45, 0.40], [0.45, 0.40, 0.35], [0.40, 0.38, 0.35], [0.38, 0.33, 0.28]]

    start_time = time.time()

    for flooding_level in flooding_levels:

        experiment_name = 'flooding_level_{}_{}_{}'.format(flooding_level[0], flooding_level[1], flooding_level[2])  # experiment name: used for indicating hyper setting

        #create test subjects dict
        test_subjects_dict = dict()
        for test_subject in test_subjects:
            
            #load this subject's test data
            sub_feature_array, sub_label_array = data_loading_function(os.path.join(data_dir, 'sub_{}.csv'.format(test_subject)), num_chunk_this_window_size=num_chunk_this_window_size)
            
            sub_data_len = len(sub_label_array)
            assert sub_data_len == int(num_chunk_this_window_size/2), 'subject {} len is not {} for binary classification'.format(test_subject, int(num_chunk_this_window_size/2))
            half_sub_data_len = int(sub_data_len/2)
            print('half_sub_data_len: {}'.format(half_sub_data_len), flush=True)
            
            sub_test_feature_array = sub_feature_array[half_sub_data_len:]
            sub_test_label_array = sub_label_array[half_sub_data_len:]
            
            #convert subject's test data into dataset object
            sub_test_set = brain_data.brain_dataset(sub_test_feature_array, sub_test_label_array)

            #convert subject's test dataset object into dataloader object
            test_batch_size = len(sub_test_set)
            sub_test_loader = torch.utils.data.DataLoader(sub_test_set, batch_size=test_batch_size, shuffle=False)
            
            #create the dict for this subject: 
            #each subject's dict has: 'transformed_sub_test_feature_array', 'sub_test_label_array',
                                # 'resutl_save_subjectdir', 'resutl_save_subject_checkpointdir', 
                                # 'result_save_subject_predictiondir', 'result_save_subject_resultanalysisdir'
                                # 'result_save_subject_trainingcurvedir', 'result_save_dir', 
        
            test_subjects_dict[test_subject] = dict()

            test_subjects_dict[test_subject]['sub_test_loader'] = sub_test_loader
            test_subjects_dict[test_subject]['sub_test_label_array'] = sub_test_label_array
            

            #derived arg
            result_save_subjectdir = os.path.join(result_save_rootdir, test_subject, experiment_name)
            result_save_subject_checkpointdir = os.path.join(result_save_subjectdir, 'checkpoint')
            result_save_subject_predictionsdir = os.path.join(result_save_subjectdir, 'predictions')
            result_save_subject_resultanalysisdir = os.path.join(result_save_subjectdir, 'result_analysis')
            result_save_subject_trainingcurvedir = os.path.join(result_save_subjectdir, 'trainingcurve')

            makedir_if_not_exist(result_save_subjectdir)
            makedir_if_not_exist(result_save_subject_checkpointdir)
            makedir_if_not_exist(result_save_subject_predictionsdir)
            makedir_if_not_exist(result_save_subject_resultanalysisdir)
            makedir_if_not_exist(result_save_subject_trainingcurvedir)
            
            test_subjects_dict[test_subject]['result_save_subjectdir'] = result_save_subjectdir
            test_subjects_dict[test_subject]['result_save_subject_checkpointdir'] = result_save_subject_checkpointdir
            test_subjects_dict[test_subject]['result_save_subject_predictionsdir'] = result_save_subject_predictionsdir
            test_subjects_dict[test_subject]['result_save_subject_resultanalysisdir'] = result_save_subject_resultanalysisdir
            test_subjects_dict[test_subject]['result_save_subject_trainingcurvedir'] = result_save_subject_trainingcurvedir

            test_subjects_dict[test_subject]['result_save_dict'] = dict()
            
        sampling_points = 150
        #create model
        model = model_to_use(n_class=2, sampling_points=sampling_points, patch_length=30, dim=240, depth=6, heads=8, mlp_dim=256).to(device)

        #reload weights from restore_file is specified
        if restore_file != 'None':
            restore_path = os.path.join(os.path.join(result_save_subject_checkpointdir, restore_file))
            print('loading checkpoint: {}'.format(restore_path))
            model.load_state_dict(torch.load(restore_path, map_location=device))

        #create criterion and optimizer
        # criterion = nn.NLLLoss() #for EEGNet and DeepConvNet, use nn.NLLLoss directly, which accept integer labels
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr) #the authors used Adam instead of SGD
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = LabelSmoothing(0.1)
        optimizer = torch.optim.AdamW(model.parameters())
        lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        #training loop
        best_val_accuracy = 0.0

        epoch_train_loss = []
        epoch_train_accuracy = []
        epoch_validation_accuracy = []
        
        early_stop = EarlyStopping(mode='max', patience=20)
        early_stop.reset()

        for epoch in trange(n_epoch, desc='1-fold cross validation'):
            average_loss_this_epoch = train_one_epoch_Ours_T_chunk(model, optimizer, criterion, group_train_loader, device, epoch, flooding_level)
            val_accuracy, _, _, _ = eval_model_OursT_chunk(model, group_val_loader, device)
            train_accuracy, _, _ , _ = eval_model_OursT_chunk(model, group_train_loader, device)

            epoch_train_loss.append(average_loss_this_epoch)
            epoch_train_accuracy.append(train_accuracy)
            epoch_validation_accuracy.append(val_accuracy)

            lrStep.step()

            #update is_best flag
            is_best = val_accuracy >= best_val_accuracy
            
            if is_best:
                best_val_accuracy = val_accuracy
                
                for test_subject in test_subjects:
                    torch.save(model.state_dict(), os.path.join(test_subjects_dict[test_subject]['result_save_subject_checkpointdir'], 'best_model.statedict'))
                                            
                    inference_start_time = time.time()
                    test_accuracy, test_class_predictions, test_class_labels, test_logits = eval_model_OursT_chunk(model, test_subjects_dict[test_subject]['sub_test_loader'], device)
                    inference_end_time = time.time()
                    inference_time = inference_end_time - inference_start_time
                    
                    print('test accuracy this epoch for subject: {} is {}'.format(test_subject, test_accuracy))
                    test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_accuracy'] = test_accuracy
                    test_subjects_dict[test_subject]['result_save_dict']['bestepoch_val_accuracy'] = val_accuracy
                    test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_logits'] = test_logits.copy()
                    test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_class_predictions'] = test_class_predictions.copy()
                    test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_class_labels'] = test_class_labels.copy()

            if early_stop(val_accuracy):
                break

        for test_subject in test_subjects:
            
            #save training curve for each fold
            save_training_curves_FixedTrainValSplit('training_curve.png', test_subjects_dict[test_subject]['result_save_subject_trainingcurvedir'], epoch_train_loss, epoch_train_accuracy, epoch_validation_accuracy)
            
            #confusion matrix 
            plot_confusion_matrix(test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_class_predictions'], test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_class_labels'], confusion_matrix_figure_labels, test_subjects_dict[test_subject]['result_save_subject_resultanalysisdir'], 'test_confusion_matrix.png')

            #save the model at last epoch
            torch.save(model.state_dict(), os.path.join(test_subjects_dict[test_subject]['result_save_subject_checkpointdir'], 'last_model.statedict'))  # they both save the same best_model and last_model for every subject in the bucket


            #save result_save_dict
            save_pickle(test_subjects_dict[test_subject]['result_save_subject_predictionsdir'], 'result_save_dict.pkl', test_subjects_dict[test_subject]['result_save_dict'])

            #write performance to txt file
            write_performance_info_FixedTrainValSplit(model.state_dict(), test_subjects_dict[test_subject]['result_save_subject_resultanalysisdir'], test_subjects_dict[test_subject]['result_save_dict']['bestepoch_val_accuracy'], test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_accuracy'])

    end_time = time.time()
    total_time = end_time - start_time
    write_program_time(result_save_rootdir, total_time)
    write_inference_time(result_save_rootdir, inference_time)


if __name__=='__main__':
    
    #parse args
    args = parser.parse_args()
    
    seed = args.seed
    gpu_idx = args.gpu_idx
    data_dir = args.data_dir
    window_size = args.window_size
    result_save_rootdir = args.result_save_rootdir
    classification_task = args.classification_task
    restore_file = args.restore_file
    n_epoch = args.n_epoch
    setting = args.setting
    
    test_subjects, train_subjects, val_subjects = generic_GetTrainValTestSubjects(setting)
    
    #sanity check:
    print('data_dir: {}, type: {}'.format(data_dir, type(data_dir)))
    print('window_size: {}, type: {}'.format(window_size, type(window_size)))
    print('result_save_rootdir: {}, type: {}'.format(result_save_rootdir, type(result_save_rootdir)))
    print('classification_task: {}, type: {}'.format(classification_task, type(classification_task)))
    print('restore_file: {} type: {}'.format(restore_file, type(restore_file)))
    print('n_epoch: {} type: {}'.format(n_epoch, type(n_epoch)))
    print('setting: {} type: {}'.format(setting, type(setting)))
    
    args_dict = edict() 
    
    args_dict.gpu_idx = gpu_idx
    args_dict.data_dir = data_dir
    args_dict.window_size = window_size
    args_dict.result_save_rootdir = result_save_rootdir
    args_dict.classification_task = classification_task
    args_dict.restore_file = restore_file
    args_dict.n_epoch = n_epoch

    
    
    seed_everything(seed)
    train_classifier(args_dict, train_subjects, val_subjects, test_subjects)
    
