#!/usr/bin/env bash

python3 train_FEDet.py \
    --dataset=VOC \
    --basenet=vgg16_reducedfc.pth \
    --batch_size=32 \
    --img_dim=300 \
    --num_workers=4 \
    --start_iter=0 \
    --cuda=True \
    --lr=0.001 \
    --save_folder=weights/fedet300_voc07+12_ch256 \
    --arch="FEDet" \
    --use_dataAug=False \
    --use_aux=True \
    --use_rfm=True \
    --use_feature_fusion=True \
    #2>&1 | tee log/train_FEDet300_voc07+12_ch128.log &
