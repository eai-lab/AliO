#!/bin/bash

source ./venv/bin/activate

log_dir=logs/basic/ETTm2
mkdir -p $log_dir

max_parallel_jobs=3 # 최대 병렬 프로세스 개수

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

pred_lens=(96 192 336 720)
learning_rates=(0.001 0.001 0.01 0.1)

for random_seed in 2023 2024 2025; do
  for i in "${!pred_lens[@]}"; do
    pred_len=${pred_lens[$i]}
    learning_rate=${learning_rates[$i]}
    wait_for_available_slot  # 빈 슬롯이 생길 때까지 기다림
    model_id="ETTm2_${pred_len}"
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
      --alio_num_samples 2 \
      --alio_lag 1 \
      --alio_time_weight 1.0 \
      --alio_freq_weight 1.0 \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id $model_id \
      --model DLinear \
      --data ETTm2 \
      --features M \
      --seq_len 336 \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 32 \
      --learning_rate $learning_rate \
      --random_seed $random_seed \
      --fast_dataloader \
      $@ >> $log_path &

    sleep 15

    done
done

wait

bash scripts/exe_01.generate_result.sh

echo "Finished training ETTm2"
