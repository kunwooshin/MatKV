#!/bin/bash

num=0
sum=0
min=999999
max=0

# [1] 스크립트가 종료될 때 최종 결과를 표시하기 위한 cleanup 함수
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

# [2] SIGINT(Ctrl+C)나 SIGTERM(kill) 시 cleanup() 함수를 호출
trap cleanup INT TERM

echo "측정을 시작합니다. (중단하려면 Ctrl+C 또는 kill)"

# [3] 무한 루프 돌면서 전력 측정
while true; do
    # 1) ipmitool로 Instantaneous power reading만 뽑아오기
    reading=$(sudo ipmitool dcmi power reading | grep 'Instantaneous power reading:' | awk '{print $4}')
    if [ -z "$reading" ]; then
        echo "ipmitool에서 전력값을 읽지 못했습니다."
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

    # 3) 1초 대기 (원하면 다른 주기로 변경 가능)
    sleep 1
done
