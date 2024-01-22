#!/bin/bash

source /usr/anaconda3/etc/profile.d/conda.sh
conda activate stella_mm

CUDA_AVAILABLE_DEVICES=1 python main.py --seed 1 --layer_depth 128 --experiment_name 241022_CheXpert_layer_depth_128_seed1 --dataset CheXpert