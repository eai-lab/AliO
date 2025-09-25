#!/bin/bash

source ./venv/bin/activate

log_dir=logs/LongForecasting/PatchTST
mkdir -p $log_dir

max_parallel_jobs=1 # 최대 병렬 프로세스 개수

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

seq_len=336
model_name=PatchTST

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

for random_seed in 2023 2024; do
    for pred_len in 720 336 96 192; do
        wait_for_available_slot  # 빈 슬롯이 생길 때까지 기다림
        model_id="${model_id_name}_${seq_len}_${pred_len}"
        log_path="$log_dir/${model_id}/${random_seed}.log"
        mkdir -p $log_dir/$model_id

        echo "Start training ${model_id} with random seed ${random_seed}"

        current_time=$(date "+%Y.%m.%d-%H.%M.%S")
        echo "============================================" >> $log_path
        echo "Start training ${model_id} with random seed ${random_seed}" >> $log_path
        echo "Exp Time : $current_time" >> $log_path
        # batch 24
        python -u main.py \
        --alio_num_samples 2 \
        --alio_lag 1 \
        --alio_time_weight 1.0 \
        --alio_freq_weight 1.0 \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 862 \
        --e_layers 3 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --dropout 0.2 \
        --fc_dropout 0.2 \
        --head_dropout 0 \
        --patch_len 16 \
        --stride 8 \
        --des 'Exp' \
        --train_epochs 100 \
        --patience 10 \
        --lradj 'TST' \
        --pct_start 0.2 \
        --itr 1 \
        --batch_size 6 \
        --learning_rate 0.0001 \
        --use_amp \
        $@ >> $log_path &
    done
done

wait

echo "Finished training PatchTST LongForecasting"
