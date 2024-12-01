#!/bin/bash


for dropout in 0.1 0.2 0.3 0.4 0.5
do
  cmd="CUDA_VISIBLE_DEVICES=1 python3 fine_tuning.py \
    fine_tuning.run_name=ft_dropout_${dropout} \
    override=dista_log \
    fine_tuning.dropout=${dropout}"
  echo Issuing command: $cmd
  eval $cmd
done