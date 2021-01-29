#!/usr/bin/env bash

set -x

NAME='ast_summer2winteryosemite'
TASK='AST'
DATA='summer2winteryosemite'
CROOT='./datasets/summer2winter_yosemite'
SROOT='./datasets/summer2winter_yosemite'
CKPTROOT='./checkpoints'
WORKER=4
RESROOT='./results'
EPOCH='latest'

python test.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 0 \
    --checkpoints_dir $CKPTROOT \
    --batchSize 1 \
    --dataset_mode $DATA \
    --croot $CROOT \
    --sroot $SROOT \
    --nThreads $WORKER \
    --no_pairing_check \
    --no_instance \
    --num_upsampling_layers more \
    --alpha 1.0 \
    --results_dir $RESROOT \
    --which_epoch $EPOCH \
    --show_input
