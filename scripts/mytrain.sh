#!/bin/bash
NUM_PROC=$1
# shift
local=$(dirname "$0") # file directory path
root=$(dirname $(dirname $local)) # project path
work_dirs=/data/work_dirs/zxy/Complex_cifar10
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=$NUM_PROC ${root}/train.py \
     --dataset torch/cifar10\
     --data-dir   /data/cifar10\
     --dataset-download False\
    --output $work_dirs\
    --model resnet50 \
    --sched cosine \
    --epochs 200 \
    --warmup-epochs 5 \
    --lr 0.4 \
    --reprob 0.5 \
    --remode pixel \
    --batch-size 128 \
    -vb 128 \
    --amp -j 48 \
    --pin-mem \
    --log-interval 20 \
