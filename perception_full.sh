#!/usr/bin/env bash

labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured')
features=('faus' 'facenet512' 'vit-fer' 'w2v-msp' 'egemaps --normalize' 'ds')

# RNN
nums_rnn_layers=(2)
model_dims=(256)

# GENERAL
lrs=(0.0005)
patience=10
n_seeds=5
dropouts=(0.4)

# adapt
csv='../csvs/perception_baseline.csv'

for feature in "${features[@]}"; do
    # RNN
    for num_rnn_layers in "${nums_rnn_layers[@]}"; do
        for model_dim in "${model_dims[@]}"; do
            for lr in "${lrs[@]}";do
                for dropout in "${dropouts[@]}";do
                    for label in "${labels[@]}"; do
                        python3 main.py --task perception --use_gpu --feature $feature --model_dim $model_dim --label_dim "$label" --rnn_n_layers $num_rnn_layers --lr "$lr" --n_seeds "$n_seeds" --result_csv "$csv" --linear_dropout $dropout --rnn_dropout $dropout --early_stopping_patience 10
                    done
                    done
                done
            done
        done
    done

