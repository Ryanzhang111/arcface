#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

# --num_class 10575 \
    # --boundaries '10,16,19' \

python train.py \
    --restore '' \
    --finetune '' \
    --dataset_root ./data/webface-lcnn-40-new \
    --logdir log/resnet_18_half_wb \
    --num_class 10575 \
    --log_every 100 \
    --start_epoch 0 \
    --end_epoch 39 \
    --model resnet_18_half \
    --batch_size 128 \
    --weight_decay 1e-4 \
    --values '1e-1,1e-2,1e-3,1e-4' \
    --boundaries '16,32,36' \
    --lr_scale 1 \
    --pre_lr_scale 1 \
    --s 32 \
    --m 0.3 \
    --train_model True \
    --load_fc False \
    --n_gpu 2
