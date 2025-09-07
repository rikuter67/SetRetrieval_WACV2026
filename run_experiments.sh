#!/bin/bash

# =================================================================
#                 実験設定 (ここを編集)
# =================================================================

# --- 固定する基本設定 ---
DATASET="DeepFurniture"
FIXED_ARGS="--dataset $DATASET \
    --mode train \
    --data_dir ./data \
    --dataset_dir ./datasets \
    --use_weighted_topk \
    --candidate_neg_num 50 \
    --batch_size 64 \
    --epochs 500 \
    --seed 42 \
    --taneg_curriculum_epochs 100 \
    --learning_rate 1e-4"

# --- スイープするハイパーパラメータ ---
# (実行時間が長くなりすぎないように、組み合わせを絞っています。自由に追加・変更してください)

# TaNegのγ（ガンマ）の範囲 ("初期値 最終値")
GAMMA_RANGES=(
    "0.2 0.4"
)

# 温度 (temperature) - まずは固定して他の影響を見る
TEMPERATURES=(0.8)

# PaNegのε (epsilon) - まずは固定
EPSILONS=(0.2)

# Cycle Lossのλ (lambda)
CYCLE_LAMBDAS=(0.2)


# =================================================================
#                     4手法比較実験の実行
# =================================================================

# # --- 手法1: ベースライン (Cycle Loss無し, TPaNeg無し) ---
echo "#################################################################"
echo "### STARTING METHOD 1: BASELINE (No Cycle Loss, No TPaNeg)    ###"
echo "#################################################################"

COMMAND="TF_XLA_FLAGS=--tf_xla_auto_jit=0 CUDA_VISIBLE_DEVICES=0 python run.py \
    $FIXED_ARGS"

echo "================================================================="
echo ">>> Running Method 1 (Baseline): No additional components"
echo "================================================================="
eval $COMMAND


# # --- 手法2: Cycle Loss のみ ---
# echo "#################################################################"
# echo "### STARTING METHOD 2: Cycle Loss ON, TPaNeg OFF              ###"
# echo "#################################################################"
# for lambda in "${CYCLE_LAMBDAS[@]}"; do
    
#     # コマンド組み立て
#     COMMAND="TF_XLA_FLAGS=--tf_xla_auto_jit=0 CUDA_VISIBLE_DEVICES=0 python run.py \
#         $FIXED_ARGS \
#         --use_cycle_loss \
#         --cycle_lambda $lambda"

#     # 実行
#     echo "================================================================="
#     echo ">>> Running Method 2 | Cycle Lambda = $lambda"
#     echo "================================================================="
#     eval $COMMAND
# done


# --- 手法3: TPaNeg のみ ---
# echo "#################################################################"
# echo "### STARTING METHOD 3: Cycle Loss OFF, TPaNeg ON              ###"
# echo "#################################################################"
# for range in "${GAMMA_RANGES[@]}"; do
#     read -r -a gamma_pair <<< "$range"
#     gamma_init=${gamma_pair[0]}
#     gamma_final=${gamma_pair[1]}
#     for temp in "${TEMPERATURES[@]}"; do
#         for eps in "${EPSILONS[@]}"; do
            
#             # コマンド組み立て
#             COMMAND="TF_XLA_FLAGS=--tf_xla_auto_jit=0 CUDA_VISIBLE_DEVICES=0 python run.py \
#                 $FIXED_ARGS \
#                 --use_tpaneg \
#                 --taneg_t_gamma_init $gamma_init \
#                 --taneg_t_gamma_final $gamma_final \
#                 --temperature $temp \
#                 --paneg_epsilon $eps"
            
#             # 実行
#             echo "================================================================="
#             echo ">>> Running Method 3 | Gamma=$gamma_init->$gamma_final, Temp=$temp, Epsilon=$eps"
#             echo "================================================================="
#             eval $COMMAND
#         done
#     done
# done


# # --- 手法4: Cycle Loss + TPaNeg (提案手法) ---
# echo "#################################################################"
# echo "### STARTING METHOD 4: Cycle Loss ON, TPaNeg ON (PROPOSED)    ###"
# echo "#################################################################"
# for lambda in "${CYCLE_LAMBDAS[@]}"; do
#     for range in "${GAMMA_RANGES[@]}"; do
#         read -r -a gamma_pair <<< "$range"
#         gamma_init=${gamma_pair[0]}
#         gamma_final=${gamma_pair[1]}
#         for temp in "${TEMPERATURES[@]}"; do
#             for eps in "${EPSILONS[@]}"; do
                
#                 # コマンド組み立て
#                 COMMAND="TF_XLA_FLAGS=--tf_xla_auto_jit=0 CUDA_VISIBLE_DEVICES=0 python run.py \
#                     $FIXED_ARGS \
#                     --use_tpaneg \
#                     --use_cycle_loss \
#                     --taneg_t_gamma_init $gamma_init \
#                     --taneg_t_gamma_final $gamma_final \
#                     --temperature $temp \
#                     --paneg_epsilon $eps \
#                     --cycle_lambda $lambda"
                
#                 # 実行
#                 echo "================================================================="
#                 echo ">>> Running Method 4 (PROPOSED) | Lambda=$lambda, Gamma=$gamma_init->$gamma_final, Temp=$temp, Epsilon=$eps"
#                 echo "================================================================="
#                 eval $COMMAND
#             done
#         done
#     done
# done


# =================================================================
#                     実験完了後の処理
# =================================================================
echo "✅ All 4 methods comparison experiments finished."
echo ""
echo "=== EXPERIMENT SUMMARY ==="
echo "Method 1: Baseline (No additional components)"
echo "Method 2: Cycle Loss only (λ=${CYCLE_LAMBDAS[*]})"
echo "Method 3: TPaNeg only (γ=${GAMMA_RANGES[*]}, T=${TEMPERATURES[*]}, ε=${EPSILONS[*]})"  
echo "Method 4: Cycle Loss + TPaNeg (PROPOSED)"
echo ""
echo "🔍 Check the results in the output logs and saved model directories."
echo "📊 Compare the evaluation metrics across all 4 methods."