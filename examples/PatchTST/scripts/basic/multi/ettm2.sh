#!/bin/bash

source ./venv/bin/activate

log_dir=logs/basic/ETTm2
mkdir -p $log_dir

max_parallel_jobs=2  # 최대 병렬 프로세스 개수

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

seq_len=336
model_name=PatchTST
root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

for random_seed in 2023 2024 2025; do
    for pred_len in 96 192 336 720; do
        wait_for_available_slot  # 빈 슬롯이 생길 때까지 기다림
        model_id="${model_id_name}_${pred_len}"
        log_path="$log_dir/${model_id}/${random_seed}.log"
        mkdir -p $log_dir/$model_id

        echo "Start training ${model_id} with random seed ${random_seed}"

        current_time=$(date "+%Y.%m.%d-%H.%M.%S")
        echo "============================================" >> $log_path
        echo "Start training ${model_id} with random seed ${random_seed}" >> $log_path
        echo "Exp Time : $current_time" >> $log_path
        python -u main.py \
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
        --enc_in 7 \
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
        --patience 20 \
        --lradj 'TST' \
        --pct_start 0.4 \
        --itr 1 \
        --batch_size 128 \
        --learning_rate 0.0001  \
        $@ >> $log_path &
    done
done

# 모든 백그라운드 작업이 끝날 때까지 기다림
wait

echo "Finished training ETTm2"
