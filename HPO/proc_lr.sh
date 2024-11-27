#!/bin/bash


for processor_lr in 0.00001 0.0001 0.0005 0.00005
do
  cmd="python3 fine_tuning.py \
    fine_tuning.run_name=ft_proclr_${processor_lr} \
    override=retrain_proc \
    fine_tuning.processor_lr=${processor_lr}"
  echo Issuing command: $cmd
  eval $cmd
done
