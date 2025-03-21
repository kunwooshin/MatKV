#!/bin/bash

torchrun --nproc_per_node 1 --master_port=29500 eval_pp.py --db_dir=/mnt/raid0/kunwooshin/data_k_aio/db_70b --cache_dir=/mnt/raid0/kunwooshin/data_k_aio/cache_70b --query_file=./questions/query.jsonl --top_k 2 --use_past_cache=True --bsz 8 --max_new_tokens 20 --total_num 256 2>&1 | tee ./tmp/eval_pp_log.txt &
 
py_pid=$!

echo "[eval_pp.sh] waiting for the model to be loaded"
while true; do
    if grep -q "MODEL LOADED" ./tmp/eval_pp_log.txt; 
	then
        echo "[eval_pp.sh] model loaded. start power monitoring"

        ./power_monitor-smi.sh &
        monitor_pid=$!
        break
    fi

    if ! ps -p $py_pid > /dev/null 2>&1; then
        echo "[eval_batch.sh] eval_pp.py terminated without any monitoring (no model loaded)"
        exit 1
    fi
	

    sleep 0.5
done

echo "[eval_pp.sh] waiting for eval_pp.py to be terminated"
wait $py_pid

echo "[eval_pp.sh] all process related to eval_pp.py terminated!"


kill $monitor_pid
echo "[eval_pp.sh] power_monitor.sh ended"
