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

export YOUR_PATH="/home/jingyitao/workspace/fNIRS_models/code_data_tufts"

scenarios=("64vs4" "16vs4" "4vs4")
testsubjects=(86 56 72 79 93 82 55 48 80 14 58 75 62 47 52 84 73 69 42 63 81 15 57 70 27 92 38 76 45 24 36 71 91 85 61 83 94 31 43 54 51 64 68 44 20 32 5 49 65 28 78 37 97 40 74 46 22 7 23 95 13 35 1 34 21 25 29 60)
testbuckets=("TestBucket1" "TestBucket2" "TestBucket3" "TestBucket4" "TestBucket5" "TestBucket6" "TestBucket7" "TestBucket8" "TestBucket9" "TestBucket10" "TestBucket11" "TestBucket12" "TestBucket13" "TestBucket14" "TestBucket15" "TestBucket16" "TestBucket17")

for ((i=0; i<1; i++))
do
    export scenario=${scenarios[i]}
    for ((j=0; j<17; j++))
    do
        export TestBucket=${testbuckets[j]}
        for ((k=0; k<4; k++))
        do
            export SubjectId_of_interest=${testsubjects[4*j+k]}
            export experiment_dir="$YOUR_PATH/fNIRS-mental-workload-classifiers/experiments/generic_finetuning_models/OursT_1_3_2_1/binary/train_100/$scenario/$TestBucket/$SubjectId_of_interest"
            
            echo "Current experiment_dir is $experiment_dir"
            
            ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file
            
            bash $YOUR_PATH/fNIRS-mental-workload-classifiers/synthesizing_results/generic_models/binary/OursT/synthesize_hypersearch_OursT_for_a_subject.slurm
            
        done
    done
done