#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python nav_data_generation.py --group_num 8 --group_id 0 > log_gpu0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python nav_data_generation.py --group_num 8 --group_id 1 > log_gpu1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python nav_data_generation.py --group_num 8 --group_id 2 > log_gpu2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python nav_data_generation.py --group_num 8 --group_id 3 > log_gpu3.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 python nav_data_generation.py --group_num 8 --group_id 4 > log_gpu4.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 python nav_data_generation.py --group_num 8 --group_id 5 > log_gpu5.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 python nav_data_generation.py --group_num 8 --group_id 6 > log_gpu6.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 python nav_data_generation.py --group_num 8 --group_id 7 > log_gpu7.txt 2>&1 &