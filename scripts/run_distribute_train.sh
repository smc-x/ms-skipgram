#!/bin/bash
if [ $# != 2 ]
then
    echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]."
    exit 1
fi

export RANK_TABLE_FILE=$1
export DEVICE_NUM=8
export RANK_SIZE=8

for((i=0; i<${RANK_SIZE}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=${i}
    echo "start distributed training for rank $RANK_ID, device $DEVICE_ID"
    python train.py --device_id=$DEVICE_ID --train_data_dir=$2 --run_distribute=True &>train$i.log & 
done