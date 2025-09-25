#!/bin/bash

# Python virtual environment 활성화
source ./venv/bin/activate

# 로그 디렉토리 생성
log_dir=logs/basic/Solar
mkdir -p $log_dir

# 최대 병렬 프로세스 개수
max_parallel_jobs=3

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

# pred_len, d_model, d_ff 리스트
pred_lens=(96 192 336 720)
d_models=(512 512 512 512)  # 각 pred_len에 대한 d_model
d_ffs=(512 512 512 512)      # 각 pred_len에 대한 d_ff

# 반복문을 사용하여 각 pred_len과 random_seed에 대해 학습 수행
for random_seed in 2023 2024 2025; do
  for i in "${!pred_lens[@]}"; do

    wait_for_available_slot
    
    pred_len=${pred_lens[$i]}
    d_model=${d_models[$i]}
    d_ff=${d_ffs[$i]}
    
    model_id="solar_${pred_len}"
    log_path="$log_dir/${model_id}/${random_seed}.log"
    mkdir -p $log_dir/$model_id

    echo "Start training ${model_id} with random seed ${random_seed}"

    current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    
    # 로그 파일에 정보 기록
    echo "============================================" >> $log_path
    echo "Start training ${model_id} with random seed ${random_seed}" >> $log_path
    echo "Exp Time : $current_time" >> $log_path
    
    # Python 스크립트 실행
    python -u main.py \
      --alio_num_samples 2 \
      --alio_lag 1 \
      --alio_time_weight 1.0 \
      --alio_freq_weight 1.0 \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/Solar/ \
      --data_path solar_AL.txt \
      --model_id $model_id \
      --model iTransformer \
      --data Solar \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 2 \
      --enc_in 137 \
      --dec_in 137 \
      --c_out 137 \
      --des 'Exp' \
      --d_model $d_model \
      --d_ff $d_ff \
      --learning_rate 0.0005 \
      --itr 1 \
      $@ >> $log_path &

    sleep 30
    # Check if the training is successful
    # if [ $? -ne 0 ]; then
    #   echo "An error occurred in training ${model_id} with random seed ${random_seed}"
    #   exit 1
    # fi
  done
done

wait

echo "Finished training Solar models"
