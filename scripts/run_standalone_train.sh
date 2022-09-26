#!/bin/bash
export DEVICE_ID=0
export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

if [ $# == 2 ];
then
    python train.py --device_target=$1 --train_data_dir=$2 > train.log 2>&1 &
else
    echo "Usage: bash run_standalone_train.sh [DEVICE_TARGET] [TRAIN_DATA_DIR]."
fi