#!/bin/bash

source activate ccnf 

python train_eval_ccnf.py --exp_name electricity --conf_file_path ./config/electricity.yaml --gpu 1