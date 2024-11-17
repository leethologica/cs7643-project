#!/bin/bash

python train.py \
    --savepath ./artifacts/best_mobilenetv2.pt \
    --cocopath /u00/data/coco-2014 \
    --cocofakepath /u00/data/cocofake \
    --train-lim 10000 \
    --val-lim 5000 \
    --n-epochs 30 \
    --batch 4 \
    --num-fake 1 \
    --lr 0.001 \
    --patience 10 \
