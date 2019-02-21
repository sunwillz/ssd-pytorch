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
    --save_folder=weights/FEDet300_all0.25_VOC0712 \
    --use_dataAug=True \
    --use_aux=True \
    2>&1 | tee log/FEDet300_all0.25_VOC0712.log &
