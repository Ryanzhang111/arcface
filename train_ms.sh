#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

# --num_class 10575 \
    # --boundaries '10,16,19' \

python train.py \
    --restore '' \
    --finetune '' \
    --dataset_root ./data/MsCeleb-tfrecord \
    --logdir log/mobilefacenet_ms \
    --num_class 79077 \
    --log_every 100 \
    --start_epoch 0 \
    --end_epoch 8 \
    --model mobilefacenet \
    --batch_size 128 \
    --weight_decay 4e-5 \
    --values '1e-1,1e-2,1e-3,1e-4' \
    --boundaries '4,6,7' \
    --lr_scale 1 \
    --pre_lr_scale 1 \
    --s 64 \
    --m 0.5 \
    --train_model True \
    --load_fc False \
    --n_gpu 2
