#!/bin/bash

# torchrun --nproc_per_node 1 --master_port=29500 eval_batch.py --db_dir=/mnt/raid0/kunwooshin/data/db --cache_dir=/mnt/raid0/kunwooshin/data/cache --query_file=./questions/query.jsonl --top_k 2 --use_past_cache=True --bsz 32 --max_new_tokens 30 --total_num 128

torchrun --nproc_per_node 1 --master_port=29500 eval_batch.py --db_dir=/mnt/raid0/kunwooshin/data_k/db_8b --cache_dir=/mnt/raid0/kunwooshin/data_k/cache_8b --query_file=./questions/query.jsonl --top_k 2 --use_past_cache=True --bsz 1 --max_new_tokens 20 --total_num 128