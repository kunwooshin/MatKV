#!/bin/bash

torchrun --nproc_per_node 1 --master_port=29500 eval_pp.py --db_dir=/mnt/raid0/kunwooshin/data_k_aio/db_70b --cache_dir=/mnt/raid0/kunwooshin/data_k_aio/cache_70b --query_file=./questions/query.jsonl --top_k 2 --use_past_cache=True --bsz 8 --max_new_tokens 20 --total_num 256 2>&1 | tee ./tmp/eval_pp_log.txt &
 
py_pid=$!

# 2) 로그에서 "MODEL LOADED" 문구가 나올 때까지 감시
echo "[eval_pp.sh] 모델 로딩 감시 중..."
while true; do
    # 'MODEL LOADED'가 로그에 찍혔는지 확인
    if grep -q "MODEL LOADED" ./tmp/eval_pp_log.txt; 
	then
        echo "[eval_pp.sh] 모델 로딩 완료 감지! 전력 모니터 시작!"

        # 3) power_monitor.sh 같은 스크립트를 실행
        ./power_monitor-smi.sh &
        monitor_pid=$!
        break
    fi

    # 만약 Python 프로세스( eval_batch.py )가 끝났으면, 감시 중단
    if ! ps -p $py_pid > /dev/null 2>&1; then
        echo "[eval_batch.sh] eval_pp.py 종료됨. 모델 로딩 문구를 찾지 못했습니다."
        exit 1
    fi
	
	# eval_pp.py 종료 시 자식 프로세스까지 정리
	# if ps -p $py_pid > /dev/null 2>&1; then
	# 	echo "[eval_pp.sh] eval_pp.py 강제 종료!"
	# 	pkill -P $py_pid  # 자식 프로세스 종료
	# 	kill -9 $py_pid   # 부모 프로세스 강제 종료
	# fi

    sleep 0.5
done

# 4) eval_batch.py( torchrun )가 끝날 때까지 대기
echo "[eval_batch.sh] eval_pp.py가 끝나길 기다립니다..."
wait $py_pid
# 모든 관련 프로세스 종료 대기
echo "[eval_pp.sh] 모든 eval_pp.py 관련 프로세스 종료 완료!"

# echo "[eval_batch.sh] eval_pp.py 종료 완료!"

# 5) 원하는 시점에 power_monitor.sh도 종료
kill $monitor_pid
echo "[eval_batch.sh] 전력 모니터링 스크립트도 종료!"
