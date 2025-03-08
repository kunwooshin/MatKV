#!/bin/bash

torchrun --nproc_per_node 1 --master_port=29500 eval_batch.py --db_dir=/mnt/raid0/kunwooshin/data/db --cache_dir=/mnt/raid0/kunwooshin/data/cache --query_file=./questions/query.jsonl --top_k 2 --use_past_cache=False --bsz 1 --max_new_tokens 100 --total_num 128
