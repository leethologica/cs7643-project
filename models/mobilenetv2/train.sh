#!/bin/bash

python train.py \
    --savepath ./artifacts/best_mobilenetv2_pretrained_false.pt \
    --cocopath /u00/data/coco-2014 \
    --cocofakepath /u00/data/cocofake \
    --train-lim 10000 \
    --val-lim 5000 \
    --n-epochs 150 \
    --batch 4 \
    --num-fake 1 \
    --lr 0.001 \
    --patience 5 \

