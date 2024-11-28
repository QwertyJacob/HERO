#!/bin/bash

# encoder HPO

for encoder in mobilenet_v3_small mobilenet_v3_large squeezenet1_0 squeezenet1_1 resnet18, 
do
  cmd="CUDA_VISIBLE_DEVICES=0 python3 fine_tuning.py \
    fine_tuning.run_name=enc_${encoder} \
    override=log \
    fine_tuning.encoder=${encoder}"
  echo Issuing command: $cmd
  eval $cmd
done

