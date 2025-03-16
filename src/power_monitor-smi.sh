#!/bin/bash

num=0
sum=0
min=999999
max=0

# [1] 스크립트 종료 시 결과 표시 함수
cleanup() {
    if [ $num -gt 0 ]; then
        avg=$((sum / num))
    else
        avg=0
    fi

    echo
    echo "측정 종료. (수집된 샘플 $num 개)"
    echo "Min: $min W"
    echo "Max: $max W"
    echo "Avg: $avg W"

    exit 0
}

# [2] SIGINT(Ctrl+C) 또는 SIGTERM(kill) 시 cleanup() 함수 호출
trap cleanup INT TERM

echo "GPU 전력 측정을 시작합니다. (중단하려면 Ctrl+C 또는 kill)"

# [3] 무한 루프 실행
while true; do
    # 1) GPU 전력 소비량(W) 가져오기
    reading=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --id=0| awk '{print int($1)}')
    
    if [ -z "$reading" ]; then
        echo "nvidia-smi에서 전력값을 읽지 못했습니다."
        cleanup  # 즉시 종료하며 결과 표시
    fi

    # 2) min, max, sum 갱신
    if [ "$reading" -lt "$min" ]; then
        min=$reading
    fi
    if [ "$reading" -gt "$max" ]; then
        max=$reading
    fi
    sum=$((sum + reading))
    num=$((num + 1))

    # 3) 1초 대기
    sleep 1
done