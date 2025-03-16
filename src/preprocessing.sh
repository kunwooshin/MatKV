#!/bin/bash

torchrun --nproc_per_node 1 preprocessing.py --docs_dir=/mnt/kunwooshin/SSD-RAG/documents --db_dir=/mnt/kunwooshin/data_k_aio/db_8b --cache_dir=/mnt/kunwooshin/data_k_aio/cache_8b --chunk_size 1024