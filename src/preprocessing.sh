#!/bin/bash

torchrun --nproc_per_node 1 preprocessing.py --docs_dir=./data/TurboRAG/documents --db_dir=./data/TurboRAG/db --cache_dir=./data/TurboRAG/cache --chunk_size 1024
