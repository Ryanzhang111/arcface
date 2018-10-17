#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# --num_class 10575 \
    # --boundaries '10,16,19' \
phrase="train"
model="resnet_456_s"
n_gpu=4
lr_scale=1

if [ "$phrase" = "train" ]; then
    nohup python train.py \
        --restore '' \
        --finetune '' \
        --dataset_root ./data/msra_celeb_id40_all \
        --logdir log/${model}_msra_celeb_id40_all_v1 \
        --num_class 566586 \
        --log_every 200 \
        --start_epoch 0 \
        --end_epoch 14 \
        --model $model \
        --batch_size 40 \
        --weight_decay 1e-4 \
        --values '1e-1,1e-2,1e-3,1e-4' \
        --boundaries '6,10,12' \
        --lr_scale $lr_scale \
        --pre_lr_scale 1 \
        --s 64 \
        --m 0.5 \
        --train_model True \
        --n_epoch_softmax 2 \
        --n_gpu $n_gpu |& tee tmp.log &
elif [ "$phrase" = "finetune" ]; then
    nohup python train.py \
        --restore '' \
        --finetune './log/resnet_456_s_msra_celeb_v1/resnet_456_s_12-543075' \
        --dataset_root ./data/msra_celeb_id40 \
        --model $model \
        --logdir log/${model}_msra_celeb_v1_ft \
        --num_class 538858 \
        --log_every 200 \
        --start_epoch 0 \
        --end_epoch 8 \
        --batch_size 40 \
        --weight_decay 1e-4 \
        --values '1e-1,1e-2,1e-3,1e-4' \
        --boundaries '3,6,7' \
        --lr_scale $lr_scale \
        --pre_lr_scale 1 \
        --s 64 \
        --m 0.5 \
        --train_model True \
        --n_epoch_softmax 1 \
        --n_gpu "$n_gpu" |& tee tmp.log &
fi
