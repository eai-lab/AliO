#!/bin/bash

# 가상환경 활성화
source ./venv/bin/activate

# 로그 디렉토리 설정
log_dir=logs/basic/ETTm1
mkdir -p $log_dir

# 최대 병렬 프로세스 개수 설정
max_parallel_jobs=1

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

# 설정된 랜덤 시드와 예측 길이에 대해 반복 실행
for random_seed in 2023 2024 2025; do
  for pred_len in 24 48 96 288 672; do
    wait_for_available_slot  # 빈 슬롯이 생길 때까지 대기
    model_id="ETTm1_${pred_len}"
    log_path="$log_dir/${model_id}/${random_seed}.log"
    mkdir -p $log_dir/$model_id

    echo "Start training ${model_id} with random seed ${random_seed}"

    current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    echo "============================================" >> $log_path
    echo "Start training ${model_id} with random seed ${random_seed}" >> $log_path
    echo "Exp Time : $current_time" >> $log_path

    # Autoformer 모델 실행
    python -u main.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm1.csv \
      --model_id $model_id \
      --model Autoformer \
      --data ETTm1 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --freq 't' \
      --itr 1 \
      --random_seed $random_seed \
      $@ >> $log_path &

  done
done

wait

echo "Finished training ETTm1"
