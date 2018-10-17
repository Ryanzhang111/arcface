#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# --num_class 10575 \
    # --boundaries '10,16,19' \

python train.py \
    --restore '' \
    --finetune './log/resnet_456_s_ms/resnet_456_s_11-371364' \
    --dataset_root ./data/ms-id-40w-aug \
    --logdir log/resnet_456_s_ms_id40w_aug_ft \
    --num_class 482564 \
    --log_every 100 \
    --start_epoch 0 \
    --end_epoch 6 \
    --model resnet_456_s \
    --batch_size 40 \
    --weight_decay 5e-4 \
    --values '1e-1,1e-2,1e-3,1e-4' \
    --boundaries '2,4,5' \
    --lr_scale 0.05 \
    --pre_lr_scale 0.1 \
    --s 64 \
    --m 0.5 \
    --train_model True \
    --n_gpu 4
