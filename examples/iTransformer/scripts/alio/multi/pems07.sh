#!/bin/bash

source ./venv/bin/activate

log_dir=logs/alio/PEMS07
mkdir -p $log_dir

# pred_len, d_model, d_ff 리스트
pred_lens=(12 24 48 96)
d_models=(512 512 512 512)  # 각 pred_len에 대한 d_model
d_ffs=(512 512 512 512)      # 각 pred_len에 대한 d_ff
e_layers=(2 2 4 4)
batch_sizes=(32 32 16 16)

for random_seed in 2023 2024 2025; do
  for i in "${!pred_lens[@]}"; do
    pred_len=${pred_lens[$i]}
    d_model=${d_models[$i]}
    d_ff=${d_ffs[$i]}
    e_layer=${e_layers[$i]}
    batch_size=${batch_sizes[$i]}
    
    model_id="PEMS07-${pred_len}"
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
      --data_path PEMS07.npz \
      --model_id $model_id \
      --model iTransformer \
      --data PEMS \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers $e_layer \
      --enc_in 883 \
      --dec_in 883 \
      --c_out 883 \
      --des 'Exp' \
      --d_model $d_model \
      --d_ff $d_ff \
      --learning_rate 0.001 \
      --batch_size $batch_size \
      --use_norm 0 \
      --itr 1  \
      $@ >> $log_path
    
    # Check if the training is successful
    # if [ $? -ne 0 ]; then
    #   echo "An error occurred in training ${model_id} with random seed ${random_seed}"
    #   exit 1
    # fi
  done
done

echo "Finished training PEMS07"
