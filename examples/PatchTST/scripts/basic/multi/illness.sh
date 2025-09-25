#!/bin/bash

source ./venv/bin/activate

log_dir=logs/basic/illness
mkdir -p $log_dir

for random_seed in 2023 2024 2025; do
    for pred_len in 24 36 48 60; do
        model_id="national_illness_${pred_len}"
        log_path="$log_dir/${model_id}/${random_seed}.log"
        mkdir -p $log_dir/$model_id

        echo "Start training ${model_id} with random seed ${random_seed}"

        current_time=$(date "+%Y.%m.%d-%H.%M.%S")
        echo "============================================" >> $log_path
        echo "============================================" >> $log_path
        echo "============================================" >> $log_path
        echo "Start training ${model_id} with random seed ${random_seed}" >> $log_path
        echo "Exp Time : $current_time" >> $log_path
        python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path illness.csv \
        --model_id $model_id \
        --model PatchTST \
        --data custom \
        --features M \
        --seq_len 104 \
        --pred_len $pred_len \
        --enc_in 7 \
        --e_layers 3 \
        --n_heads 4 \
        --d_model 16 \
        --d_ff 128 \
        --dropout 0.3 \
        --fc_dropout 0.3 \
        --head_dropout 0 \
        --patch_len 24 \
        --stride 2 \
        --des 'Exp' \
        --train_epochs 100 \
        --lradj 'constant' \
        --itr 1 \
        --batch_size 16 \
        --learning_rate 0.0025 >> $log_path
    done
done

echo "Finished training illness"
