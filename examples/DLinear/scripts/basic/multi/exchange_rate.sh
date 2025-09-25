#!/bin/bash

source ./venv/bin/activate

log_dir=logs/basic/exchange
mkdir -p $log_dir

seq_len=336
max_parallel_jobs=3 # 최대 병렬 프로세스 개수

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

# 예측 길이에 따른 batch_size와 learning_rate 리스트 설정
pred_lens=(96 192 336 720)
batch_sizes=(8 8 32 32)
learning_rates=(0.0005 0.0005 0.0005 0.005)

for random_seed in 2023 2024 2025; do
    for i in ${!pred_lens[@]}; do
        wait_for_available_slot  # 빈 슬롯이 생길 때까지 기다림

        pred_len=${pred_lens[$i]}
        batch_size=${batch_sizes[$i]}
        learning_rate=${learning_rates[$i]}

        model_id="exchange_${pred_len}"
        log_path="$log_dir/${model_id}.log"
        mkdir -p $log_dir/$model_id

        echo "Start training ${model_id}"

        current_time=$(date "+%Y.%m.%d-%H.%M.%S")
        echo "============================================" >> $log_path
        echo "============================================" >> $log_path
        echo "============================================" >> $log_path
        echo "Start training ${model_id}" >> $log_path
        echo "Exp Time : $current_time" >> $log_path

        python -u main.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path exchange_rate.csv \
            --model_id $model_id \
            --model DLinear \
            --data custom \
            --features M \
            --seq_len 336 \
            --pred_len $pred_len \
            --enc_in 8 \
            --des 'Exp' \
            --itr 1 \
            --batch_size $batch_size \
            --learning_rate $learning_rate \
            --random_seed $random_seed \
            $@ >> $log_path &

    done
done

wait

echo "Finished training Exchange"
