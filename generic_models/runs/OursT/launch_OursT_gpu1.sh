#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either "list" or "submit" or "run_here"

if [[ -z $1 ]]; then
    ACTION_NAME="list"
else
    ACTION_NAME=$1
fi

export YOUR_PATH="/home/jyt/workspace/fNIRS_models/code_data_tufts"
export gpu_idx=0
export data_dir="$YOUR_PATH/fNIRS2MW/experiment/fNIRS_data/band_pass_filtered/slide_window_data/size_30sec_150ts_stride_03ts/"
export window_size=150
export classification_task="binary"
export scenario="64vs4"
export n_epoch=60


buckets=("TestBucket5" "TestBucket6" "TestBucket7" "TestBucket8")
settings64vs4=("64vs4_TestBucket5" "64vs4_TestBucket6" "64vs4_TestBucket7" "64vs4_TestBucket8")
settings16vs4=("16vs4_TestBucket5" "16vs4_TestBucket6" "16vs4_TestBucket7" "16vs4_TestBucket8")
settings4vs4=("4vs4_TestBucket5" "4vs4_TestBucket6" "4vs4_TestBucket7" "4vs4_TestBucket8")

for ((i=0; i<4; i++))
do 
    export bucket=${buckets[i]}
    export setting=${settings64vs4[i]}
    export result_save_rootdir="$YOUR_PATH/fNIRS-mental-workload-classifiers/experiments/generic_models/OursT_1_finetune/binary/$scenario/$bucket" 
    export restore_file="$YOUR_PATH/fNIRS-mental-workload-classifiers/experiments/generic_models/OursT_1_pre/binary/$scenario/$bucket" 

    bash $YOUR_PATH/fNIRS-mental-workload-classifiers/generic_models/runs/do_experiment_OursT.slurm
done
