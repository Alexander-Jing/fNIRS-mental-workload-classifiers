import os
import numpy as np
import csv
import argparse
import pandas as pd

def main(experiment_dir, summary_save_dir):
    # calculate the summary of all subjects, should run the synthesis of every subject first 

    AllSubject_summary_filename = os.path.join(summary_save_dir, 'AllSubjects_summary.csv')
    
    with open(AllSubject_summary_filename, mode='w') as csv_file:
        
        fieldnames = ['subject_id', 'max_validation_accuracy', 'corresponding_test_accuracy', 'performance_string', 'experiment_folder']
        
        fileEmpty = os.stat(AllSubject_summary_filename).st_size == 0
        
        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
        
        if fileEmpty:
            writer.writeheader()
        
        """subject_list = [ 4, 41, 69,  3, 15, 52, 42, 38, 34, 66, 35, 24, 40, 26, 16, 80, 27,
        73, 20, 12, 11, 67, 94, 44, 92, 75,  5, 59, 71, 28, 47, 85, 68, 55,
        60, 91, 84, 21, 37, 56, 36, 10, 83, 93, 81, 29,  7, 74, 86, 25, 79,
        76, 18, 48, 95,  1,  8, 61, 51, 70, 17, 64, 62, 49,  9, 72, 45, 43,
        63, 14, 19,  2, 57, 82, 53, 54, 46, 97, 22, 50, 32, 78, 30, 31, 23,
        58, 65, 13]"""
        
        subject_list = [ 1, 13, 14, 15, 20, 21, 22, 23, 24, 25, 
        27, 28, 29, 31, 32, 34, 35, 36, 37, 38, 
        40, 42, 43, 44, 45, 46, 47, 48, 49, 5, 
        51, 52, 54, 55, 56, 57, 58, 60, 61, 62,
        63, 64, 65, 68, 69, 7,  70, 71, 72, 73,
        74, 75, 76, 78, 79, 80, 81, 82, 83, 84,
        85, 86, 91, 92, 93, 94, 95, 97]

        for subject_id in subject_list:
            this_subject_summary_csv_path = os.path.join(experiment_dir, str(subject_id), 'hypersearch_summary/hypersearch_summary.csv')
            
            this_subject_summary_df = pd.read_csv(this_subject_summary_csv_path)
            
            this_subject_selected_setting = this_subject_summary_df.sort_values(by=['validation_accuracy'], ascending=False).iloc[0]
            
            this_subject_dict = {}
            this_subject_max_validation_accuracy = this_subject_selected_setting.validation_accuracy
            this_subject_corresponding_test_accuracy = this_subject_selected_setting.test_accuracy
            this_subject_performance_string = this_subject_selected_setting.performance_string
            this_subject_experiment_folder = this_subject_selected_setting.experiment_folder
            
            this_subject_dict.update(subject_id=subject_id, max_validation_accuracy=this_subject_max_validation_accuracy, corresponding_test_accuracy=this_subject_corresponding_test_accuracy, performance_string=this_subject_performance_string, experiment_folder=this_subject_experiment_folder)
            
            writer.writerow(this_subject_dict)
            

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir')
    
    #parse args
    args = parser.parse_args()
    
    experiment_dir = args.experiment_dir
    summary_save_dir = experiment_dir + '_summary'
    
    if not os.path.exists(summary_save_dir):
        os.makedirs(summary_save_dir)
        
    main(experiment_dir, summary_save_dir)
    
    
    
    
    
        
        