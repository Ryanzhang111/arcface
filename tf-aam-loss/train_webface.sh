#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# --num_class 10575 \
    # --boundaries '10,16,19' \

python train.py \
    --restore './log/resnet_456_s_ms/resnet_456_s_0-30947' \
    --dataset_root ./data/MsCeleb-tfrecord \
    --logdir log/resnet_456_s_ms \
    --num_class 79077 \
    --log_every 200 \
    --start_epoch 1 \
    --end_epoch 15 \
    --model resnet_456_s \
    --batch_size 40 \
    --weight_decay 5e-4 \
    --values '1e-1,1e-2,1e-3,1e-4' \
    --boundaries '6,10,13' \
    --lr_scale 0.5 \
    --pre_lr_scale 1. \
    --s 64 \
    --m 0.5 \
    --train_model True \
    --load_fc True \
    --n_gpu 4
