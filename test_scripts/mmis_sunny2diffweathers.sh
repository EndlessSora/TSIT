#!/usr/bin/env bash

set -x

NAME='mmis_sunny2diffweathers'
TASK='MMIS'
DATA='sunny2diffweathers'
CROOT='./datasets/bdd100k'
SROOT='./datasets/bdd100k'
CKPTROOT='./checkpoints'
WORKER=4
RESROOT='./results'
EPOCH='latest'
MODE=${1:-'all'}

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
    --show_input \
    --test_mode $MODE
