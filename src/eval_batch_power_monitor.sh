#!/bin/bash

torchrun --nproc_per_node=1 --master_port=29500 eval_batch.py \
    --db_dir=/mnt/raid0/kunwooshin/data_k_aio/db_70b \
    --cache_dir=/mnt/raid0/kunwooshin/data_k_aio/cache_70b \
    --query_file=./questions/query.jsonl \
    --top_k=2 \
    --use_past_cache=False \
    --bsz=8 \
    --max_new_tokens=20 \
    --total_num=256 \
    2>&1 | tee /tmp/eval_batch_log.txt &

py_pid=$!

echo "[eval_batch.sh] waiting for model load..."
while true; do
    if grep -q "MODEL LOADED" /tmp/eval_batch_log.txt; then
        echo "[eval_batch.sh] model loaded. start monitoring power"

        ./power_monitor-ipmi.sh & #./power_monitor-smi.sh for GPU power consumption monitoring
        monitor_pid=$!
        break
    fi

    if ! ps -p $py_pid > /dev/null 2>&1; then
        echo "[eval_batch.sh] eval_batch.py ended without any monitoring (no model loaded)."
        exit 1
    fi

    sleep 0.5
done

echo "[eval_batch.sh] waiting for eval_batch.py to end"
wait $py_pid
echo "[eval_batch.sh] eval_batch.py ended!"

kill $monitor_pid
echo "[eval_batch.sh] end power_monitor script"
