#!/bin/bash


# Training a more robust processor by reducing the 
# learning rate of the decoder and or encoder

for h_dim in 512 1200 1500 64 128 256
do
  cmd="CUDA_VISIBLE_DEVICES=0 python3 pretraining.py \
    pretraining.run_name=hspace_${h_dim} \
    override=dista_log \
    pretraining.h_dim=${h_dim} \
    pretraining.save_model=true"
  echo Issuing command: $cmd
  eval $cmd

  # fine-tuning phase using such a 
  # pre-trained processor

  cmd="CUDA_VISIBLE_DEVICES=0 python3 fine_tuning.py \
    fine_tuning.run_name=ft_hspace_${h_dim} \
    override=dista_log \
    fine_tuning.pretrained_processor_1=hspace_${h_dim}_proc_1 \
    fine_tuning.pretrained_processor_2=hspace_${h_dim}_proc_2"
  echo Issuing command: $cmd
  eval $cmd
done