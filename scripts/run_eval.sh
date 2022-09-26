#!/bin/bash
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=1
export RANK_ID=0

if [ $# == 1 ];
then
    python eval.py --eval_data_dir=$1 &> eval.log &
elif [ $# == 3 ];
then
    python eval.py --checkpoint_path=$1 --dictionary=$2 --eval_data_dir=$3 &> eval.log &
else
    echo "Usage1: sh run_eval.sh [EVAL_DATA_DIR]"
    echo "Usage2: sh run_eval.sh [CHECKPOINT_PATH] [ID2WORD_DICTIONARY] [EVAL_DATA_DIR]"
fi 