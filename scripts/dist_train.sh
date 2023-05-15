#!/bin/bash
NUM_PROC=$1
# shift
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path
work_dirs=/data/work_dirs/wyh/timm_train
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=$NUM_PROC ${root}/train.py \
     /data/imagenet\
    --train-split ILSVRC2012_img_train \
    --val-split val \
    --output $work_dirs\
    --model resnet34 \
    --sched cosine \
    --epochs 150 \
    --warmup-epochs 5 \
    --lr 0.4 \
    --reprob 0.5 \
    --remode pixel \
    --batch-size 256 \
    -vb 1024 \
    --amp -j 48 \
    --pin-mem \
    --log-interval 100 \
