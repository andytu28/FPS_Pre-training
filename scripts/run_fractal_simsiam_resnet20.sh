#!/bin/bash


GPU=$1
EXPAND=$2

CUDA_VISIBLE_DEVICES=$GPU python main_simsiam.py --data_name fractal \
  --arch_name resnet20 --expand $EXPAND --data_file ifs-1mil.pkl \
  --num_workers 4 --num_class 100000 --num_systems 100000 --max_epoch 100 \
  --max_num_objs 2 --color_mode random --gen_num_augs 2 --image_level_augs
