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

for SubjectId_of_interest in 1 13 14 15 20 21 22 23 24 25 27 28 29 31 32 34 35 36 37 38 40 42 43 44 45 46 47 48 49 5 51 52 54 55 56 57 58 60 61 62 63 64 65 68 69 7 70 71 72 73 74 75 76 78 79 80 81 82 83 84 85 86 91 92 93 94 95 97
do
    export experiment_dir="$YOUR_PATH/fNIRS-mental-workload-classifiers/experiments/subject_specific_models/DeepConvNet/binary/window_size150/$SubjectId_of_interest"
    
    echo "Current experiment_dir is $experiment_dir"
    
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < $YOUR_PATH/fNIRS-mental-workload-classifiers/synthesizing_results/subject_specific_models/binary/DeepConvNet/synthesize_hypersearch_DeepConvNet_for_a_subject.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash $YOUR_PATH/fNIRS-mental-workload-classifiers/synthesizing_results/subject_specific_models/binary/DeepConvNet/synthesize_hypersearch_DeepConvNet_for_a_subject.slurm
    fi
    
done
