#!/bin/bash -l
#$ -P ec523
#$ -N vggt_aggregator_3
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=12:00:00
#$ -l gpu_memory=40G
#$ -j y
#$ -o /projectnb/ec523/projects/proj_vggt/EC523-project/logs/qsub_aggregator_3.log

cd /projectnb/ec523/projects/proj_vggt/EC523-project
/projectnb/ec523/projects/proj_vggt/env/bin/torchrun --nproc_per_node=1 training/launch_frozen_vggt_aggregator_3.py
