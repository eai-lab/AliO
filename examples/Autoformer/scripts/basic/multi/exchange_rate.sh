#!/bin/bash

# 가상환경 활성화
source ./venv/bin/activate

# 로그 디렉토리 설정
log_dir=logs/exchange_rate
mkdir -p $log_dir

# 최대 병렬 프로세스 개수 설정
max_parallel_jobs=2

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

GPU_COUNT=0

# 설정된 랜덤 시드와 예측 길이에 대해 반복 실행
for random_seed in 2023 2024 2025; do
  # 예측 길이에 대해 반복 실행
  for pred_len in 96 192 336 720; do
    wait_for_available_slot  # 빈 슬롯이 생길 때까지 대기
    model_id="Exchange_${pred_len}"
    log_path="$log_dir/${model_id}.log"
    mkdir -p $log_dir/$model_id

    echo "Start training ${model_id}"

    current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    echo "============================================" >> $log_path
    echo "Start training ${model_id}" >> $log_path
    echo "Exp Time : $current_time" >> $log_path

    # train_epochs 옵션 처리 (pred_len에 따라 다르게 설정)
    if [ $pred_len -eq 96 ] || [ $pred_len -eq 720 ]; then
      train_epochs_option=""
    else
      train_epochs_option="--train_epochs 1"
    fi


    GPU_ID=$(( GPU_COUNT % 2 ))
    GPU_ID=$(( GPU_ID + 2 ))

    export CUDA_VISIBLE_DEVICES=$GPU_ID

    # Autoformer 모델 실행
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/exchange_rate/ \
        --data_path exchange_rate.csv \
        --model_id $model_id \
        --model Autoformer \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --des 'Exp' \
        --random_seed $random_seed \
        --itr 1 \
        $train_epochs_option \
        $@ >> $log_path &
    GPU_COUNT=$(( GPU_COUNT + 1 ))
  done
done
wait

echo "Finished training for all pred_len values"
