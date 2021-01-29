#!/usr/bin/env bash

set -x

NAME='sis_ade20k'
TASK='SIS'
DATA='ade20k'
CROOT='./datasets/ADEChallengeData2016'
SROOT='./datasets/ADEChallengeData2016'
CKPTROOT='./checkpoints'
WORKER=4

python train.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 0,1 \
    --checkpoints_dir $CKPTROOT \
    --batchSize 16 \
    --dataset_mode $DATA \
    --croot $CROOT \
    --sroot $SROOT \
    --nThreads $WORKER \
    --gan_mode hinge \
    --num_upsampling_layers more \
    --use_vae \
    --alpha 1.0 \
    --display_freq 5000 \
    --save_epoch_freq 20 \
    --niter 100 \
    --niter_decay 100 \
    --lambda_vgg 20 \
    --lambda_feat 10
