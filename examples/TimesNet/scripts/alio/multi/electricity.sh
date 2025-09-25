#!/bin/bash

source ./venv/bin/activate

log_dir=logs/basic/electricity
mkdir -p $log_dir

max_parallel_jobs=1 # 최대 병렬 프로세스 개수

# 현재 실행 중인 프로세스 개수를 체크하는 함수
function wait_for_available_slot {
  while [ $(jobs -rp | wc -l) -ge $max_parallel_jobs ]; do
    sleep 1
  done
}

# pred_len, d_model, d_ff를 리스트로 정의
pred_len_list=(96 192 336 720)
d_model_list=(256 256 256 256)
d_ff_list=(512 512 512 512)

for random_seed in 2023 2024 2025; do
    for i in "${!pred_len_list[@]}"; do
        wait_for_available_slot  # 빈 슬롯이 생길 때까지 기다림
        pred_len=${pred_len_list[$i]}
        d_model=${d_model_list[$i]}
        d_ff=${d_ff_list[$i]}

        model_id="ECL_${pred_len}"
        log_path="$log_dir/${model_id}/${random_seed}.log"
        mkdir -p $log_dir/$model_id

        echo "Start training ${model_id} with random seed ${random_seed}"

        current_time=$(date "+%Y.%m.%d-%H.%M.%S")
        echo "============================================" >> $log_path
        echo "Start training ${model_id} with random seed ${random_seed}" >> $log_path
        echo "Exp Time : $current_time" >> $log_path

        use_amp=""
        if [ $d_model -eq 720 ]; then
            use_amp="--use_amp"
        fi

        python -u main.py \
            --alio_num_samples 2 \
            --alio_lag 1 \
            --alio_time_weight 1.0 \
            --alio_freq_weight 1.0 \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./dataset/electricity/ \
            --data_path electricity.csv \
            --model_id $model_id \
            --model TimesNet \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --d_model $d_model \
            --d_ff $d_ff \
            --top_k 5 \
            --des 'Exp' \
            --itr 1 \
            --random_seed $random_seed \
            $use_amp \
            $@ >> $log_path &

    done
done

wait

echo "Finished training electricity dataset"
