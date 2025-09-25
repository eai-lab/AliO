#!/bin/bash

# 가상환경 활성화
source ./venv/bin/activate

# 로그 저장 경로
log_dir=logs/basic/electricity
mkdir -p $log_dir

# pred_len, d_model, d_ff 리스트
pred_lens=(96 192 336 720)
d_models=(512 512 512 512)  # 각 pred_len에 대한 d_model
d_ffs=(512 512 512 512)      # 각 pred_len에 대한 d_ff

# 여러 random seed 값에 대해 실험 반복
for random_seed in 2023 2024 2025; do
  for i in "${!pred_lens[@]}"; do
    pred_len=${pred_lens[$i]}
    d_model=${d_models[$i]}
    d_ff=${d_ffs[$i]}
    
    # 모델 ID 생성
    model_id="electricity_${pred_len}"
    log_path="$log_dir/${model_id}/${random_seed}.log"
    mkdir -p $log_dir/$model_id

    echo "Start training ${model_id} with random seed ${random_seed}"

    # 현재 시간 출력
    current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    echo "============================================" >> $log_path
    echo "Start training ${model_id} with random seed ${random_seed}" >> $log_path
    echo "Experiment start time: $current_time" >> $log_path

    # Python 스크립트 실행
    python -u main.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id $model_id \
      --model iTransformer \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size 16 \
      --learning_rate 0.0005 \
      --itr 1 >> $log_path

    # 훈련 성공 여부 확인
    # if [ $? -ne 0 ]; then
    #   echo "An error occurred in training ${model_id} with random seed ${random_seed}"
    #   exit 1
    # fi
  done
done

echo "Finished training for electricity dataset"
