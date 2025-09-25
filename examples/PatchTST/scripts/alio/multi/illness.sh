#!/bin/bash

source ./venv/bin/activate

log_dir=logs/alio_awl/illness
mkdir -p $log_dir

max_parallel_jobs=1 # 최대 병렬 프로세스 개수

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

for random_seed in 2023 2024 2025; do
    for pred_len in 24 36 48 60; do
        wait_for_available_slot  # 빈 슬롯이 생길 때까지 기다림
        model_id="illness_${pred_len}"
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
        --random_seed $random_seed \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path illness.csv \
        --model_id $model_id \
        --model PatchTST \
        --data custom \
        --features M \
        --seq_len 104 \
        --pred_len $pred_len \
        --enc_in 7 \
        --e_layers 3 \
        --n_heads 4 \
        --d_model 16 \
        --d_ff 128 \
        --dropout 0.3 \
        --fc_dropout 0.3 \
        --head_dropout 0 \
        --patch_len 24 \
        --stride 2 \
        --des 'Exp' \
        --train_epochs 100 \
        --lradj 'constant' \
        --itr 1 \
        --batch_size 16 \
        --learning_rate 0.0025 \
        $@ >> $log_path &
    done
done

wait

echo "Finished training illness"
