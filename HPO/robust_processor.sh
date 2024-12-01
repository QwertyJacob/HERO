#!/bin/bash


# Training a more robust processor by reducing the 
# learning rate of the decoder and or encoder

cmd="CUDA_VISIBLE_DEVICES=0 python3 pretraining.py \
  pretraining.run_name=pt_enc_dec_lr_0.0001 \
  override=dista_log \
  pretraining.dec_lr=0.0001 \
  pretraining.enc_lr=0.0001 \
  pretraining.save_model=true"
echo Issuing command: $cmd
eval $cmd

# fine-tuning phase using such a 
# pre-trained processor

cmd="CUDA_VISIBLE_DEVICES=0 python3 fine_tuning.py \
  fine_tuning.run_name=ft_hyper_robust_proc \
  override=dista_log \
  fine_tuning.pretrained_processor_1=pt_enc_dec_lr_0.0001_proc_1 \
  fine_tuning.pretrained_processor_2=pt_enc_dec_lr_0.0001_proc_2 \
  fine_tuning.retrain_processors=True"
echo Issuing command: $cmd
eval $cmd