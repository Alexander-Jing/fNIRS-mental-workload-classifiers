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


for SubjectId_of_interest in 22 70 78 28 60 58 36 85 34 82 29 20 55 76 56 24 13 93 
do
    export experiment_dir="/cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental-workload-classifiers1/experiments/generic_models/WhiteSubset/EEGNet/binary/21vs6/seed3/$SubjectId_of_interest"
    
    echo "Current experiment_dir is $experiment_dir"
    
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < /cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental_workload-classifiers/synthesizing_results/generic_models_White/binary/EEGNet/synthesize_hypersearch_EEGNet_for_a_subject.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash /cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental_workload-classifiers/synthesizing_results/generic_models_White/binary/EEGNet/synthesize_hypersearch_EEGNet_for_a_subject.slurm
    fi
    
done
