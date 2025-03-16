#!/bin/bash
#SBATCH --job-name llama-ssd # your job name here
#SBATCH --gres=gpu:1 # if you need 4 GPUs, fixit to 4
#SBATCH --partition=lsw
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# YOUR SCRIPT GOES HERE
#source ~/miniconda3/etc/profile.d/conda.sh 
#conda activate base

#export PATH="/home/n0/gihwan/miniconda3/bin:$PATH"  # commented out by conda initialize

torchrun --nproc_per_node 1 --master_port=29500 eval_pp.py --db_dir=/mnt/raid0/kunwooshin/data_k_aio/db_8b --cache_dir=/mnt/raid0/kunwooshin/data_k_aio/cache_8b --query_file=./questions/query.jsonl --top_k 2 --use_past_cache=True --bsz 32 --max_new_tokens 20 --total_num 256
#torchrun --nproc_per_node 1 eval.py --db_dir=data/db \
	#--cache_dir=data/cache --query_file=./questions/query.jsonl --top_k=2 --use_past_cache=True

