#!/bin/bash

# 가상환경 활성화 (필요할 경우 설정)
source ./venv/bin/activate

# 로그 디렉토리 생성
log_dir=logs/electricity
mkdir -p $log_dir

# 병렬 실행 개수 제한
max_parallel_jobs=1

# 현재 실행 중인 프로세스 개수를 확인하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

# 예측 길이별로 실행
for random_seed in 2023 2024 2025; do
  # 예측 길이에 따른 실행 반복
  for pred_len in 96 192 336 720; do
    wait_for_available_slot  # 빈 슬롯이 생길 때까지 대기

    # 모델 ID 및 로그 파일 경로 설정
    model_id="ECL_96_${pred_len}"
    log_path="$log_dir/${model_id}/${random_seed}.log"
    mkdir -p $log_dir/$model_id

    echo "Start training ${model_id} with random seed ${random_seed}"

    current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    echo "============================================" >> $log_path
    echo "Start training ${model_id} with random seed ${random_seed}" >> $log_path
    echo "Exp Time : $current_time" >> $log_path

    # 모델 실행
    python -u main.py \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
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
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --itr 1 \
      --random_seed $random_seed \
      $@ >> $log_path &

  done
done

wait
echo "All training jobs for electricity dataset have been completed."
