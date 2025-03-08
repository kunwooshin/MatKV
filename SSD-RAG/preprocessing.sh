#!/bin/bash

torchrun --nproc_per_node 1 preprocessing.py --docs_dir=/mnt/kunwooshin/SSD-RAG/documents --db_dir=/mnt/raid0/kunwooshin/data/db --cache_dir=/mnt/raid0/kunwooshin/data/cache_K --chunk_size 1024
