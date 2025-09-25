#!/bin/bash

source ./venv/bin/activate

log_dir=logs/basic/weather
mkdir -p $log_dir

# 최대 병렬 프로세스 개수
max_parallel_jobs=2

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
e_layers=(3 3 3 3)
seq_len=96
enc_in=21
dec_in=21
c_out=21

for random_seed in 2023 2024 2025; do
  for i in "${!pred_lens[@]}"; do
    wait_for_available_slot  # 빈 슬롯이 생길 때까지 기다림

    pred_len=${pred_lens[$i]}
    d_model=${d_models[$i]}
    d_ff=${d_ffs[$i]}
    e_layer=${e_layers[$i]}
    
    model_id="weather-${pred_len}"
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
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id $model_id \
      --model iTransformer \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers $e_layer \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des 'Exp' \
      --d_model $d_model \
      --d_ff $d_ff \
      --itr 1 \
      $@ >> $log_path &

  done
done

wait

echo "Finished training weather"
