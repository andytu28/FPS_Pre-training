#!/bin/bash


WORLD_SIZE=1
GPU=$1
PORTNUM=$2 
ARCH=resnet18
EPOCHS=100
BATCH_SIZE=256
TARGET_DATASET=multifractal_224_cls-100000_sys-100000_mnobjs-5_genaugs-2_colormode-random_w-imgaug_wo-bg

DIST_URL='tcp://localhost:'$PORTNUM


SLURM_PROCID=0 CUDA_VISIBLE_DEVICES=$GPU python main_moco_distributed.py --lr 0.03 --mlp --moco-t 0.2 --aug-plus --cos \
  --arch $ARCH --batch-size $BATCH_SIZE --epochs $EPOCHS --world-size $WORLD_SIZE \
  --target-dataset $TARGET_DATASET --dist-url $DIST_URL --multiprocessing-distributed \
  --world-size $WORLD_SIZE --rank -1 --save-freq 10 --ngpus-per-node 1 \
  --dist-backend nccl NONE
