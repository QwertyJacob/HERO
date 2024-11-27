#!/bin/bash

# We should test pretraining the processor 
# with different (small) learning rates IN THE ENCODER.
for enc_dec_lr in 0.00001 0.0001 0.00005 0.000001
do
  cmd="CUDA_VISIBLE_DEVICES=1 python3 pretraining.py \
    pretraining.run_name=pr_enc_dec_llr_${enc_dec_lr} \
    override=log \
    pretraining.enc_dec_lr=${enc_dec_lr} \
    pretraining.save_model=true"
  echo Issuing command: $cmd
  eval $cmd
done
