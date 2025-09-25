#!/bin/bash

source ./venv/bin/activate

log_dir=logs/basic/weather
mkdir -p $log_dir

max_parallel_jobs=2 # 최대 병렬 프로세스 개수

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

GPU_COUNT=0

for random_seed in 2023 2024 2025; do
    for pred_len in 96 192 336 720; do
        wait_for_available_slot  # 빈 슬롯이 생길 때까지 기다림
        
        model_id="weather_${pred_len}"
        log_path="$log_dir/${model_id}/${random_seed}.log"
        mkdir -p $log_dir/$model_id

        echo "Start training ${model_id} with random seed ${random_seed}"

        GPU_ID=$((GPU_COUNT % 2))
        GPU_ID=$((GPU_ID + 2))

        export CUDA_VISIBLE_DEVICES=$GPU_ID
        GPU_COUNT=$((GPU_COUNT + 1))

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
            --root_path ./dataset/ \
            --data_path weather.csv \
            --model_id $model_id \
            --data custom \
            --seq_len 512 \
            --label_len 48 \
            --pred_len $pred_len \
            --batch_size 512 \
            --learning_rate 0.0001 \
            --train_epochs 10 \
            --decay_fac 0.9 \
            --d_model 768 \
            --n_heads 4 \
            --d_ff 768 \
            --dropout 0.3 \
            --enc_in 7 \
            --c_out 7 \
            --freq 0 \
            --lradj type3 \
            --patch_size 16 \
            --stride 8 \
            --percent 100 \
            --gpt_layer 6 \
            --itr 1 \
            --model GPT4TS \
            --is_gpt 1 \
            --random_seed $random_seed \
            $@ >> $log_path &

    done
done

wait

echo "Finished training weather"
