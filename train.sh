#!/bin/bash -l

#Specify project
#$ -P ec523 

#Request appropriate time (default 12 hours; gpu jobs time limit - 2 days (48 hours), cpu jobs - 30 days (720 hours) )
#$ -l h_rt=12:00:00

#Send an email when the job is done or aborted (by default no email is sent)
#$ -m e

# Give job a name
#$ -N train_text 

# Join output and error streams into one file
#$ -j y


#load appropriate envornment
module load miniconda
module load pandoc 
module load texlive 

#execute the program
export HF_HOME=$(realpath ../hf_cache)
conda activate my_env && torchrun training/launch_vggt_textonly.py
