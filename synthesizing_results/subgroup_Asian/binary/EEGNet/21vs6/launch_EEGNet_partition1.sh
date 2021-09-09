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


for SubjectId_of_interest in 22 70 78 28 60 58 85 38 42 29 48 40 5 76 93 49 94 35
do
    export experiment_dir="/cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental-workload-classifiers/experiments/generic_models/AsianSubset/EEGNet/binary/21vs6/seed2/$SubjectId_of_interest"
    
    echo "Current experiment_dir is $experiment_dir"
    
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < /cluster/tufts/hugheslab/zhuang12/general_utilities/synthesize_hyperparameter_search_results/NuripsDataSet_FixedTrainValSplit/generic_models_Asian/binary/EEGNet/synthesize_hypersearch_EEGNet_for_a_subject.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash /cluster/tufts/hugheslab/zhuang12/general_utilities/synthesize_hyperparameter_search_results/NuripsDataSet_FixedTrainValSplit/generic_models_Asian/binary/EEGNet/synthesize_hypersearch_EEGNet_for_a_subject.slurm
    fi
    
done
