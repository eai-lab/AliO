#!/bin/bash

source ./venv/bin/activate

log_dir=logs/basic/traffic
mkdir -p $log_dir

max_parallel_jobs=1 # 최대 병렬 프로세스 개수

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

seq_len=336
pred_lens=(96 192 336 720)

for random_seed in 2023 2024 2025; do
    for pred_len in "${pred_lens[@]}"; do
        wait_for_available_slot  # 빈 슬롯이 생길 때까지 기다림
        model_id="traffic_${pred_len}"
        log_path="$log_dir/${model_id}.log"
        mkdir -p $log_dir

        echo "Start training ${model_id} with random seed ${random_seed}"

        current_time=$(date "+%Y.%m.%d-%H.%M.%S")
        echo "============================================" >> $log_path
        echo "Start training ${model_id} with random seed ${random_seed}" >> $log_path
        echo "Exp Time : $current_time" >> $log_path

        python -u main.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path traffic.csv \
            --model_id $model_id \
            --model DLinear \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 862 \
            --des 'Exp' \
            --itr 1 \
            --batch_size 16 \
            --learning_rate 0.05 \
            --random_seed $random_seed \
            >> $log_path &

    done
done

wait

bash scripts/exe_01.generate_result.sh

echo "Finished training traffic"
