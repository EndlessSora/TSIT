#!/usr/bin/env bash

set -x

NAME='sis_cityscapes'
TASK='SIS'
DATA='cityscapes'
CROOT='./datasets/Cityscapes'
SROOT='./datasets/Cityscapes'
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
    --num_upsampling_layers more \
    --use_vae \
    --alpha 1.0 \
    --results_dir $RESROOT \
    --which_epoch $EPOCH \
    --show_input
