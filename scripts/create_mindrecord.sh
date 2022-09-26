#!/bin/bash
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=1
export RANK_ID=0

if [ $# == 1 ];
then
    python preprocess.py --train_data_dir=$1 &> preprocess.log & 
else
    echo "Usage: sh create_mindrecord.sh [TRAIN_DATA_DIR]"
fi 