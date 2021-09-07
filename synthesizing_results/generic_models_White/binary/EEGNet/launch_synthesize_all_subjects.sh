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


for experiment in seed1 seed2 seed3 seed4 seed5 seed6 seed7
do
    export experiment_dir="/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/experiments/FixedTrainValSplit_generic_models/WhiteSubset/EEGNet/binary/21vs6/$experiment"
    
    echo "Current experiment_dir is $experiment_dir"
    
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < /cluster/tufts/hugheslab/zhuang12/general_utilities/synthesize_hyperparameter_search_results/NuripsDataSet_FixedTrainValSplit/generic_models_White/binary/EEGNet/synthesize_all_subjects.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash /cluster/tufts/hugheslab/zhuang12/general_utilities/synthesize_hyperparameter_search_results/NuripsDataSet_FixedTrainValSplit/generic_models_White/binary/EEGNet/synthesize_all_subjects.slurm
    fi
    
done