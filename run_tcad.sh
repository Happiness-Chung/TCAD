#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python main.py --experiment_name 240507_CheXpert_resnet_layer_depth_128_seed1_changed_separation_loss --model ResNet --seed 1 --layer_depth 128 --dataset CheXpert