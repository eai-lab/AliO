#!/bin/bash

source ./venv/bin/activate

log_dir=logs/alio/PEMS04
mkdir -p $log_dir

# pred_len, d_model, d_ff 리스트
pred_lens=(12 24 48 96)
d_models=(1024 1024 1024 1024)  # 각 pred_len에 대한 d_model
d_ffs=(1024 1024 1024 1024)      # 각 pred_len에 대한 d_ff

max_parallel_jobs=1 # 최대 병렬 프로세스 개수

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

for random_seed in 2023 2024 2025; do
  for i in "${!pred_lens[@]}"; do
    wait_for_available_slot  # 빈 슬롯이 생길 때까지 기다림
    pred_len=${pred_lens[$i]}
    d_model=${d_models[$i]}
    d_ff=${d_ffs[$i]}
    
    model_id="PEMS04-${pred_len}"
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
      --root_path ./dataset/PEMS/ \
      --data_path PEMS04.npz \
      --model_id $model_id \
      --model iTransformer \
      --data PEMS \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 4 \
      --enc_in 307 \
      --dec_in 307 \
      --c_out 307 \
      --des 'Exp' \
      --d_model $d_model \
      --d_ff $d_ff \
      --learning_rate 0.0005 \
      --use_norm 0 \
      --itr 1 \
      $@ >> $log_path &
    
    # Check if the training is successful
    # if [ $? -ne 0 ]; then
    #   echo "An error occurred in training ${model_id} with random seed ${random_seed}"
    #   exit 1
    # fi
  done
done

# 모든 백그라운드 작업이 끝날 때까지 기다림
wait

echo "Finished training PEMS04"
