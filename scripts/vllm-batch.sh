#!/bin/bash

PROMPTS_FILE=batch-prompts.jsonl
BATCHES="1 3 5 10 20 50 100 151"
PROMPTS="100 500 1000 2000 5000 7000 10000 15000 20000 30000"
OUTPUTS="1 50 100 150 200"

PRINT_HEADER=1
for output_count in ${OUTPUTS}; do
    for batch_size in ${BATCHES}; do
        for prompt_size in ${PROMPTS}; do
            echo "### batch=${batch_size} prompt-size=${prompt_size} output-count=${output_count}"
            CUDA_VISIBLE_DEVICES=0 python vllm-batch.py vllm-batch ${PROMPTS_FILE} ${batch_size} ${prompt_size} ${output_count} $PRINT_HEADER
            PRINT_HEADER=0
        done
    done
done

# EOF
