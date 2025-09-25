#!/bin/bash

# Python 환경 활성화
source ./venv/bin/activate

# 로그 디렉토리 설정
log_dir="logs/basic/exchange_rate"
mkdir -p $log_dir

# 최대 병렬 실행 개수
max_parallel_jobs=1

# 실행 가능한 슬롯 확인 함수
function wait_for_available_slot {
    while [ "$(jobs -rp | wc -l)" -ge $max_parallel_jobs ]; do
        sleep 1
    done
}

# 실험 변수 설정
pred_len_list=(96 192 336 720)
d_model_list=(64 64 32 32)
d_ff_list=(64 64 32 32)
random_seeds=(2023 2024 2025)

# 학습 루프
for random_seed in "${random_seeds[@]}"; do
    for i in "${!pred_len_list[@]}"; do
        wait_for_available_slot
        
        # 변수 설정
        pred_len=${pred_len_list[$i]}
        d_model=${d_model_list[$i]}
        d_ff=${d_ff_list[$i]}
        model_id="Traffic_${pred_len}"
        log_path="$log_dir/${model_id}/${random_seed}.log"
        mkdir -p "$(dirname "$log_path")"

        # 로그 시작
        echo "Starting training for $model_id with seed $random_seed"
        echo "============================================"
        echo "Training $model_id with seed $random_seed"
        echo "Experiment Time: $(date '+%Y.%m.%d-%H.%M.%S')"

        # 학습 실행
        python -u main.py \
            --alio_num_samples 2 \
            --alio_lag 1 \
            --alio_time_weight 1.0 \
            --alio_freq_weight 1.0 \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./dataset/traffic/ \
            --data_path traffic.csv \
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
            --enc_in 862 \
            --dec_in 862 \
            --c_out 862 \
            --d_model 512 \
            --d_ff 512 \
            --top_k 5 \
            --des 'Exp' \
            --itr 1 \
            --batch_size 16 \
            --random_seed $random_seed \
            $@ &

    done
done

# 모든 작업 완료 대기
wait

echo "Finished training exchange_rate dataset"
