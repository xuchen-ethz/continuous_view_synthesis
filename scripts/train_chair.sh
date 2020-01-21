#!/bin/bash
python train.py\
    --name chair\
    --category chair\
    --niter 2000 \
    --niter_decay 2000 \
    --save_epoch_freq 100
