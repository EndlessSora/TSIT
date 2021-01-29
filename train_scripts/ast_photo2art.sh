#!/usr/bin/env bash

set -x

NAME='ast_photo2art'
TASK='AST'
DATA='photo2art'
CROOT='./datasets/photo2art'
SROOT='./datasets/photo2art'
CKPTROOT='./checkpoints'
WORKER=4

python train.py \
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
    --gan_mode hinge \
    --num_upsampling_layers more \
    --alpha 1.0 \
    --display_freq 200 \
    --save_epoch_freq 10 \
    --niter 40 \
    --lambda_vgg 1 \
    --lambda_feat 2
