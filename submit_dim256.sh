#!/bin/bash -l
#$ -P ec523
#$ -N vggt_dim256
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=12:00:00
#$ -l gpu_memory=48G
#$ -j y
#$ -o /projectnb/ec523/projects/proj_vggt/EC523-project/logs/qsub_dim256.log
cd /projectnb/ec523/projects/proj_vggt/EC523-project
/projectnb/ec523/projects/proj_vggt/env/bin/torchrun --nproc_per_node=1 --master_port=29502 training/launch_frozen_vggt_dim256.py
