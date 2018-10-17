#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

# --num_class 10575 \
    # --boundaries '10,16,19' \
phrase="train"
model="shufflenetv2"
n_gpu=2
lr_scale=1

if [ "$phrase" = "train" ]; then
    python train.py \
        --restore '' \
        --finetune '' \
        --dataset_root ./data/msra_celeb \
        --logdir log/${model}_msra_celeb_v1 \
        --num_class 180855 \
        --log_every 200 \
        --start_epoch 0 \
        --end_epoch 14 \
        --model $model \
        --batch_size 128 \
        --weight_decay 1e-4 \
        --values '1e-1,1e-2,1e-3,1e-4' \
        --boundaries '6,11,13' \
        --lr_scale $lr_scale \
        --pre_lr_scale 1 \
        --s 64 \
        --m 0.5 \
        --train_model True \
        --load_fc False \
        --n_epoch_softmax 4 \
        --n_gpu $n_gpu 
        #|& tee tmp.log &
elif [ "$phrase" = "finetune" ]; then
    python train.py \
        --restore '' \
        --finetune 'log/resnet_18_half_wb/resnet_18_half_29-51900' \
        --dataset_root ./data/msra_celeb \
        --model $model \
        --logdir log/${model}_msra_celeb_vtest \
        --num_class 180855 \
        --log_every 10 \
        --start_epoch 0 \
        --end_epoch 9 \
        --batch_size 256 \
        --weight_decay 1e-4 \
        --values '1e-1,1e-2,1e-3,1e-4' \
        --boundaries '4,7,8' \
        --lr_scale $lr_scale \
        --pre_lr_scale 1 \
        --s 64 \
        --m 0.5 \
        --train_model True \
        --n_epoch_softmax 1 \
        --n_gpu "$n_gpu"
    #|& tee tmp_1.log &
fi
