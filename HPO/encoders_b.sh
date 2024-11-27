#!/bin/bash


for encoder in squeezenet1_0, squeezenet1_1
do
  cmd="CUDA_VISIBLE_DEVICES=1 python3 fine_tuning.py \
    fine_tuning.run_name=enc_${encoder} \
    override=log \
    fine_tuning.encoder=${encoder}"
  echo Issuing command: $cmd
  eval $cmd
done
