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

export gpu_idx=0
export data_dir='/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/data/Leon/Visual/size_2sec_10ts_stride_3ts/'
export window_size=10
export result_save_rootdir='/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/experiments/FixedTrainValSplit_subject_specific_models/LogisticRegression/binary/window_size10'
export classification_task='binary'
for SubjectId_of_interest in 45 43 63 14 19 2 57 82 53 54 46 97 22 50 32 78 30 31 23 58 65 13

do
    export SubjectId_of_interest=$SubjectId_of_interest

    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < /cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/subject_specific_models/runs_FixedTrainValSplit/do_experiment_LogisticRegression_FixedTrainValSplit.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash /cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/subject_specific_models/runs_FixedTrainValSplit/do_experiment_LogisticRegression_FixedTrainValSplit.slurm
    fi

done

