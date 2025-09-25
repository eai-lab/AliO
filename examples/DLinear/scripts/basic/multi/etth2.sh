#!/bin/bash

source ./venv/bin/activate

log_dir=logs/basic/ETTh2
mkdir -p $log_dir

max_parallel_jobs=6 # 최대 병렬 프로세스 개수

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

for random_seed in 2023 2024 2025; do
    for pred_len in 96 192 336 720; do
        wait_for_available_slot  # 빈 슬롯이 생길 때까지 기다림
        model_id="ETTh2_${pred_len}"
        log_path="$log_dir/${model_id}/${random_seed}.log"
        mkdir -p $log_dir/$model_id

        echo "Start training ${model_id} with random seed ${random_seed}"

        current_time=$(date "+%Y.%m.%d-%H.%M.%S")
        echo "============================================" >> $log_path
        echo "============================================" >> $log_path
        echo "============================================" >> $log_path
        echo "Start training ${model_id} with random seed ${random_seed}" >> $log_path
        echo "Exp Time : $current_time" >> $log_path
        python -u main.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path ETTh2.csv \
            --model_id $model_id \
            --model DLinear \
            --data ETTh2 \
            --features M \
            --seq_len 336 \
            --pred_len $pred_len \
            --enc_in 7 \
            --des 'Exp' \
            --itr 1 \
            --batch_size 32 \
            --learning_rate 0.05 \
            --random_seed $random_seed \
            $@ >> $log_path &
    done
done

wait

echo "Finished training ETTh2"
