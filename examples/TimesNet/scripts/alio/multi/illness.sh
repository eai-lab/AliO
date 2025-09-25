#!/bin/bash

source ./venv/bin/activate

log_dir=logs/basic/illness
mkdir -p $log_dir

max_parallel_jobs=1 # 최대 병렬 프로세스 개수

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

# pred_len 리스트 정의
pred_len_list=(24 36 48 60)

for random_seed in 2023 2024 2025; do
    for pred_len in "${pred_len_list[@]}"; do
        wait_for_available_slot  # 빈 슬롯이 생길 때까지 기다림
        model_id="ili_36_${pred_len}"
        log_path="$log_dir/${model_id}/${random_seed}.log"
        mkdir -p $log_dir/$model_id

        echo "Start training ${model_id} with random seed ${random_seed}"

        current_time=$(date "+%Y.%m.%d-%H.%M.%S")
        echo "============================================" >> $log_path
        echo "Start training ${model_id} with random seed ${random_seed}" >> $log_path
        echo "Exp Time : $current_time" >> $log_path

        python -u main.py \
            --alio_num_samples 2 \
            --alio_lag 1 \
            --alio_time_weight 1.0 \
            --alio_freq_weight 1.0 \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./dataset/illness/ \
            --data_path illness.csv \
            --model_id $model_id \
            --model TimesNet \
            --data custom \
            --features M \
            --seq_len 36 \
            --label_len 18 \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --d_model 768 \
            --d_ff 768 \
            --des 'Exp' \
            --itr 1 \
            --top_k 5 \
            --random_seed $random_seed \
            $@ >> $log_path &

    done
done

wait

echo "Finished training illness dataset"
