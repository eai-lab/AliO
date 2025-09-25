#!/bin/bash

# Python 가상 환경 활성화
source ./venv/bin/activate

# 로그 디렉토리 생성
log_dir=logs/basic/Electricity
mkdir -p $log_dir

# 병렬 실행을 위한 최대 병렬 프로세스 개수 설정
max_parallel_jobs=1  # 병렬로 실행할 최대 작업 수

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

# 실행할 random_seed 값들
random_seed=2021
seq_len=336
model_name=PatchTST
root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

# 각 pred_len 값에 대해 학습 실행
for random_seed in 2023 2024 2025; do
  for pred_len in 720 336 192 96; do
      wait_for_available_slot  # 빈 슬롯이 생길 때까지 대기
      model_id="${model_id_name}_${pred_len}"
      log_path="$log_dir/${model_id}/${random_seed}.log"
      mkdir -p $log_dir/$model_id

      echo "Start training ${model_id} with random seed ${random_seed}"

      current_time=$(date "+%Y.%m.%d-%H.%M.%S")
      echo "============================================" >> $log_path
      echo "Start training ${model_id} with random seed ${random_seed}" >> $log_path
      echo "Exp Time : $current_time" >> $log_path

      # --batch_size 32 \
      # Python 학습 스크립트 실행
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
          --enc_in 321 \
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
          --batch_size 16 \
          --use_amp \
          --learning_rate 0.0001 \
          $@>> $log_path &  # 백그라운드에서 실행
  done
done

# 모든 백그라운드 작업이 끝날 때까지 대기
wait

echo "Finished training Electricity models"
