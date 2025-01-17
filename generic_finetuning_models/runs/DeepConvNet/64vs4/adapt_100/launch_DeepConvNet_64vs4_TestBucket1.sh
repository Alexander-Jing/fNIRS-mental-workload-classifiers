#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export YOUR_PATH="/home/jyt/workspace/fNIRS_models/code_data_tufts"
export gpu_idx=0
export data_dir="$YOUR_PATH/fNIRS2MW/experiment/fNIRS_data/band_pass_filtered/slide_window_data/size_30sec_150ts_stride_03ts/"
export window_size=150
export classification_task='binary'
export scenario='64vs4'
export bucket='TestBucket1'
export setting="64vs4_TestBucket1"
export n_epoch=300
export restore_file="$YOUR_PATH/fNIRS-mental-workload-classifiers/experiments/generic_models/DeepConvNet/binary/64vs4/TestBucket1/72/lr1.0_dropout0.25/checkpoint/best_model.statedict"
export adapt_on='train_100'
export result_save_rootdir="$YOUR_PATH/fNIRS-mental-workload-classifiers/experiments/generic_finetuning_models/DeepConvNet/binary/$adapt_on/$scenario/$bucket" 


if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < $YOUR_PATH/fNIRS-mental-workload-classifiers/generic_finetuning_models/runs/do_experiment_DeepConvNet.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash $YOUR_PATH/fNIRS-mental-workload-classifiers/generic_finetuning_models/runs/do_experiment_DeepConvNet.slurm
fi

