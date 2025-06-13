#!/bin/bash
# =================================================================
#  Simple Experiment Runner for DeepFurniture & IQON3000
# =================================================================

set -e

# --- ⚙️ Configuration ---
# 1. Datasets to test
DATASETS=("DeepFurniture" "IQON3000")

# 2. Architectures to test (Format: {Layers}L{Heads}H)
ARCHITECTURES=("1L1H" "2L2H" "4L4H")

# 3. Batch sizes to test
BATCH_SIZES=("64" "128" "256")

# 4. Seeds for repeated runs (3 trials)
SEEDS=(42 123 2025)

# 5. Static parameters
LEARNING_RATE="1e-4"
EPOCHS="200"
GPU_ID="0"

# --- 🎨 Styling ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# --- 🚀 Main Logic ---
main() {
    # Calculate the total number of experiments
    local total_experiments=$((${#DATASETS[@]} * ${#ARCHITECTURES[@]} * ${#BATCH_SIZES[@]} * ${#SEEDS[@]}))
    local exp_count=0

    echo -e "${BLUE}=====================================================${NC}"
    echo -e "${BLUE}  Starting Experiment Runner...${NC}"
    echo -e "${BLUE}  Total experiments to run: ${total_experiments}${NC}"
    echo -e "${BLUE}=====================================================${NC}"

    local start_time=$(date +%s)

    # Loop through every combination of parameters
    for dataset in "${DATASETS[@]}"; do
        for arch in "${ARCHITECTURES[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                # <<--- 3回のシード値ループを追加 ---
                for seed in "${SEEDS[@]}"; do
                    exp_count=$((exp_count + 1))
                    echo -e "\n${BLUE}--- [${exp_count}/${total_experiments}] Running Experiment ---${NC}"

                    if [[ "$arch" =~ ^([0-9]+)L([0-9]+)H$ ]]; then
                        local layers="${BASH_REMATCH[1]}"
                        local heads="${BASH_REMATCH[2]}"
                    else
                        echo -e "${RED}Error: Invalid architecture format: '$arch'. Skipping.${NC}"
                        continue
                    fi

                    echo "  - Dataset:       ${dataset}"
                    echo "  - Architecture:  ${layers} Layers, ${heads} Heads"
                    echo "  - Batch Size:    ${batch_size}"
                    echo "  - Seed:          ${seed}" # <<--- 現在のシード値を表示

                    # <<--- pythonコマンドに --seed 引数を追加 ---
                    if CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
                        --dataset "$dataset" \
                        --mode "train" \
                        --batch-size "$batch_size" \
                        --num-layers "$layers" \
                        --num-heads "$heads" \
                        --learning-rate "$LEARNING_RATE" \
                        --epochs "$EPOCHS" \
                        --seed "$seed"; then
                        echo -e "${GREEN}✓ Experiment completed successfully.${NC}"
                    else
                        echo -e "${RED}✗ Experiment FAILED. Stopping script.${NC}"
                        exit 1
                    fi
                done
            done
        done
    done

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    echo -e "\n${BLUE}=====================================================${NC}"
    echo -e "${GREEN}All experiments finished! 🎉${NC}"
    echo -e "Total time elapsed: ${hours}h ${minutes}m ${seconds}s."
    echo -e "${BLUE}=====================================================${NC}"
}

main