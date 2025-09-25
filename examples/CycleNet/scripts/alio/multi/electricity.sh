#!/bin/bash

source ./venv/bin/activate

log_dir=logs/alio/Electricity
mkdir -p $log_dir

max_parallel_jobs=2 # 최대 병렬 프로세스 개수

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

for random_seed in 2023 2024 2025; do
    for pred_len in 96 192 336 720; do
        wait_for_available_slot
        model_id="Electricity_${pred_len}"
        log_path="$log_dir/${model_id}/${random_seed}.log"
        mkdir -p $log_dir/$model_id

        current_time=$(date "+%Y-%m-%d %H:%M:%S")
        echo "============================================" >> $log_path
        echo "Start training ${model_id} with random seed ${random_seed}" >> $log_path
        echo "Exp Time : $current_time" >> $log_path

        nohup python -u run.py \
            --alio_num_sample 2 \
            --alio_lag 1 \
            --alio_freq_weight 1.0 \
            --alio_time_weight 1.0 \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path electricity.csv \
            --model_id $model_id \
            --model CycleNet \
            --data custom \
            --features M \
            --seq_len 96 \
            --pred_len $pred_len \
            --enc_in 321 \
            --cycle 168 \
            --model_type linear \
            --train_epochs 30 \
            --patience 5 \
            --itr 1 --batch_size 64 --learning_rate 0.01 --random_seed $random_seed \
            $@ > $log_path 2>&1 &
    done
done

wait

