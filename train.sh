#!/usr/bin/env bash

python3 train.py \
    --dataset=VOC \
    --dataset_root= \
    --basenet=vgg16_reducedfc.pth \
    --batch_size=32 \
    --num_workers=4 \
    --cuda=True \
    --lr=0.001 \
    --visdom \
    --save_folder=weights \
