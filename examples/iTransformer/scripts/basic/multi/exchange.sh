#!/bin/bash

source ./venv/bin/activate

log_dir=logs/basic/exchange_rate
mkdir -p $log_dir

# pred_len, d_model, d_ff 리스트
pred_lens=(96 192 336 720)
d_models=(128 128 128 128)  # 각 pred_len에 대한 d_model
d_ffs=(128 128 128 128)      # 각 pred_len에 대한 d_ff

for random_seed in 2023 2024 2025; do
  for i in "${!pred_lens[@]}"; do
    pred_len=${pred_lens[$i]}
    d_model=${d_models[$i]}
    d_ff=${d_ffs[$i]}
    
    model_id="exchange_rate_${pred_len}"
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
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/exchange_rate/ \
      --data_path exchange_rate.csv \
      --model_id $model_id \
      --model iTransformer \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 2 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --d_model $d_model \
      --d_ff $d_ff \
      --itr 1 \
      $@ >> $log_path
    
    # Check if the training is successful
    # if [ $? -ne 0 ]; then
    #   echo "An error occurred in training ${model_id} with random seed ${random_seed}"
    #   exit 1
    # fi
  done
done

echo "Finished training exchange_rate"
