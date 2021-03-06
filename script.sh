#!/bin/bash

python train_eval_ccnf.py --exp_id 0 --dataset electricity --batch_size 64 --epochs 100 \
        --batches_per_epoch 2000 --early_stopping 5 --length 168 --pred_len 24 \
        --num_flows 10 --hidden 1024 --gpu 1 --gpu_id 1
