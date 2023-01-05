#!/bin/bash
#source /users/visics/ajha/.bash_custom

echo "Current path is $PATH"
echo "Running"
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=1 python -u main.py --dataset cpv2 \
--mode gge_iter \
--debias gradient \
--topq 1 \
--topv -1 \
--qvp 5 \
--output gradient_cpv2 |tee traing_log.txt\
