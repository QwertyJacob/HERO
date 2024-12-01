#!/bin/bash

# The main ablation of the paper is w.r.t to our 
# previous work: the NERO framework, where training is done 
# on a single phase, end-to-end.

cmd="CUDA_VISIBLE_DEVICES=1 python3 fine_tuning.py \
  fine_tuning.run_name=NERO \
  override=dista_log \
  fine_tuning.retrain_processors=True \
  fine_tuning.processor_lr=0.001 \
  fine_tuning.pretrained_processors=False"
echo Issuing command: $cmd
eval $cmd