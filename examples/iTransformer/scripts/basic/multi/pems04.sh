#!/bin/bash

source ./venv/bin/activate

log_dir=logs/basic/PEMS04
mkdir -p $log_dir

# pred_len, d_model, d_ff 리스트
pred_lens=(12 24 48 96)
d_models=(1024 1024 1024 1024)  # 각 pred_len에 대한 d_model
d_ffs=(1024 1024 1024 1024)      # 각 pred_len에 대한 d_ff

for random_seed in 2023 2024 2025; do
  for i in "${!pred_lens[@]}"; do
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
      --itr 1 >> $log_path
    
    # Check if the training is successful
    # if [ $? -ne 0 ]; then
    #   echo "An error occurred in training ${model_id} with random seed ${random_seed}"
    #   exit 1
    # fi
  done
done

echo "Finished training PEMS04"
