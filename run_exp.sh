#!/bin/bash
# =================================================================
#  TPaNeg Method Experiment Runner for WACV Paper
#  Based on: "Cyclic Compatibility Transformation and Target-Prediction-aware Hard Negative Mining"
# =================================================================

set -e

# --- ⚙️ Configuration --- #
# 1. Datasets to test
DATASETS=("DeepFurniture" "IQON3000")

# 2. Architectures to test (Layers x Heads - based on paper: L=2, H=2)
ARCHITECTURES=("2L2H")

# 3. Batch sizes (paper uses 64)
BATCH_SIZES=("64")

# 4. Seeds for repeated runs (3 trials for statistical significance)
# SEEDS=(42 123 1000)
SEEDS=(42)

# 5. TPaNeg parameters (core contribution of the paper)
# TaNeg T_gamma curriculum learning parameters
TANEG_T_GAMMA_INIT_VALUES=("0.0")  # Initial similarity threshold
# TANEG_T_GAMMA_FINAL_VALUES=("0.2" "0.4") # Final similarity threshold
TANEG_T_GAMMA_FINAL_VALUES=("0.2") # Final similarity threshold

# TANEG_CURRICULUM_EPOCHS=("50" "200")     # Curriculum learning epochs
TANEG_CURRICULUM_EPOCHS=("200")     # Curriculum learning epochs

# PaNeg epsilon (margin parameter)
PANEG_EPSILON_VALUES=("0.2" "1.0")
PANEG_EPSILON_VALUES=("0.2")

# 6. Method variations (ablation study)
# METHOD_VARIATIONS=("baseline" "cycle_only" "tpaneg_only" "full_method")
METHOD_VARIATIONS=("baseline")
METHOD_VARIATIONS=("tpaneg_only")

# 7. Static parameters based on paper
LEARNING_RATE="1e-4"
TEMPERATURE_IQON="0.8"
TEMPERATURE_DEEPFURNITURE="0.8"  # Paper uses 0.5, but code default is 0.8
EARLY_STOP="10"        # Paper uses 15 for IQON3000, 50 for DeepFurniture
CANDIDATE_NEG_NUM="300" # Number of negative candidates
CYCLE_LAMBDA="0.2"     # Cycle consistency weight
GPU_ID="0"

# --- 🎨 Styling --- #
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# --- 📊 Helper Functions --- #
get_epochs_for_dataset() {
    local dataset=$1
    case $dataset in
        "IQON3000")
            echo "100"  # Paper uses 100 epochs with early stopping
            ;;
        "DeepFurniture")
            echo "500" # Paper uses 1000 epochs with early stopping
        #     ;;
        # *)
        #     echo "200"  # Default
        #     ;;
    esac
}

get_temperature_for_dataset() {
    local dataset=$1
    case $dataset in
        "IQON3000")
            echo "$TEMPERATURE_IQON"
            ;;
        "DeepFurniture")
            echo "$TEMPERATURE_DEEPFURNITURE"
            ;;
        *)
            echo "0.8"
            ;;
    esac
}

get_taneg_params_for_dataset() {
    local dataset=$1
    case $dataset in
        "IQON3000")
            echo "0.0 0.5 50"  # init_gamma final_gamma curriculum_epochs
            ;;
        "DeepFurniture")
        #     echo "0.2 0.4 300" # init_gamma final_gamma curriculum_epochs
        #     ;;
        # *)
            echo "0.0 0.4 300"
            ;;
    esac
}

# --- 🚀 Main Logic --- #
main() {
    # Calculate the total number of experiments
    local total_experiments=0
    
    for dataset in "${DATASETS[@]}"; do
        for arch in "${ARCHITECTURES[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                for method in "${METHOD_VARIATIONS[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        if [[ "$method" == "tpaneg_only" || "$method" == "full_method" ]]; then
                            # TPaNeg parameter combinations
                            for init_gamma in "${TANEG_T_GAMMA_INIT_VALUES[@]}"; do
                                for final_gamma in "${TANEG_T_GAMMA_FINAL_VALUES[@]}"; do
                                    for epsilon in "${PANEG_EPSILON_VALUES[@]}"; do
                                        total_experiments=$((total_experiments + 1))
                                    done
                                done
                            done
                        else
                            total_experiments=$((total_experiments + 1))
                        fi
                    done
                done
            done
        done
    done
    
    local exp_count=0
    
    echo -e "${BLUE}=====================================================${NC}"
    echo -e "${BLUE}  TPaNeg Method Experiment Runner (WACV Paper)${NC}"
    echo -e "${BLUE}  Total experiments to run: ${total_experiments}${NC}"
    echo -e "${BLUE}=====================================================${NC}"
    
    local start_time=$(date +%s)
    
    # Main experiment loops
    for dataset in "${DATASETS[@]}"; do
        for arch in "${ARCHITECTURES[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                for method in "${METHOD_VARIATIONS[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        
                        # Parse architecture
                        if [[ "$arch" =~ ^([0-9]+)L([0-9]+)H$ ]]; then
                            local layers="${BASH_REMATCH[1]}"
                            local heads="${BASH_REMATCH[2]}"
                        else
                            echo -e "${RED}Error: Invalid architecture format: '$arch'. Skipping.${NC}"
                            continue
                        fi
                        
                        # Get dataset-specific parameters
                        local epochs=$(get_epochs_for_dataset "$dataset")
                        local temperature=$(get_temperature_for_dataset "$dataset")
                        
                        # TPaNeg parameter handling
                        if [[ "$method" == "tpaneg_only" || "$method" == "full_method" ]]; then
                            # Run TPaNeg parameter combinations
                            for init_gamma in "${TANEG_T_GAMMA_INIT_VALUES[@]}"; do
                                for final_gamma in "${TANEG_T_GAMMA_FINAL_VALUES[@]}"; do
                                    for epsilon in "${PANEG_EPSILON_VALUES[@]}"; do
                                        exp_count=$((exp_count + 1))
                                        run_experiment "$dataset" "$layers" "$heads" "$batch_size" "$method" \
                                                     "$seed" "$epochs" "$temperature" "$init_gamma" "$final_gamma" "$epsilon" \
                                                     "$exp_count" "$total_experiments"
                                    done
                                done
                            done
                        else
                            # Run without TPaNeg parameters
                            exp_count=$((exp_count + 1))
                            run_experiment "$dataset" "$layers" "$heads" "$batch_size" "$method" \
                                         "$seed" "$epochs" "$temperature" "0.0" "0.4" "0.2" \
                                         "$exp_count" "$total_experiments"
                        fi
                    done
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

# --- 🧪 Experiment Runner Function --- #
run_experiment() {
    local dataset=$1
    local layers=$2
    local heads=$3
    local batch_size=$4
    local method=$5
    local seed=$6
    local epochs=$7
    local temperature=$8
    local init_gamma=$9
    local final_gamma=${10}
    local epsilon=${11}
    local exp_count=${12}
    local total_experiments=${13}
    
    echo -e "\n${BLUE}--- [${exp_count}/${total_experiments}] Running Experiment ---${NC}"
    echo "  - Dataset:       ${dataset}"
    echo "  - Architecture:  ${layers} Layers, ${heads} Heads"
    echo "  - Batch Size:    ${batch_size}"
    echo "  - Method:        ${method}"
    echo "  - Seed:          ${seed}"
    echo "  - Epochs:        ${epochs}"
    echo "  - Temperature:   ${temperature}"
    
    # Build command based on method variation
    local cmd="CUDA_VISIBLE_DEVICES=$GPU_ID python run.py"
    cmd+=" --dataset $dataset"
    cmd+=" --mode train"
    cmd+=" --data_dir ./data"
    cmd+=" --dataset_dir ./datasets"
    cmd+=" --batch_size $batch_size"
    cmd+=" --num_layers $layers"
    cmd+=" --num_heads $heads"
    cmd+=" --epochs $epochs"
    cmd+=" --early_stop $EARLY_STOP"
    cmd+=" --learning_rate $LEARNING_RATE"
    cmd+=" --temperature $temperature"
    cmd+=" --seed $seed"
    cmd+=" --candidate_neg_num $CANDIDATE_NEG_NUM"
    
    # Method-specific flags
    case $method in
        "baseline")
            echo "  - Configuration: Baseline CST only"
            ;;
        "cycle_only")
            echo "  - Configuration: CST + Cycle Consistency"
            cmd+=" --use_cycle_loss"
            cmd+=" --cycle_lambda $CYCLE_LAMBDA"
            ;;
        "tpaneg_only")
            echo "  - Configuration: CST + TPaNeg Mining"
            echo "  - TPaNeg T_γ:    ${init_gamma} → ${final_gamma}"
            echo "  - PaNeg ε:       ${epsilon}"
            cmd+=" --use_tpaneg"
            cmd+=" --taneg_t_gamma_init $init_gamma"
            cmd+=" --taneg_t_gamma_final $final_gamma"
            cmd+=" --taneg_curriculum_epochs 50"  # Fixed based on paper
            cmd+=" --paneg_epsilon $epsilon"
            ;;
        "full_method")
            echo "  - Configuration: Full Method (Cycle + TPaNeg)"
            echo "  - TPaNeg T_γ:    ${init_gamma} → ${final_gamma}"
            echo "  - PaNeg ε:       ${epsilon}"
            echo "  - Cycle λ:       ${CYCLE_LAMBDA}"
            cmd+=" --use_cycle_loss"
            cmd+=" --cycle_lambda $CYCLE_LAMBDA"
            cmd+=" --use_tpaneg"
            cmd+=" --taneg_t_gamma_init $init_gamma"
            cmd+=" --taneg_t_gamma_final $final_gamma"
            cmd+=" --taneg_curriculum_epochs 50"
            cmd+=" --paneg_epsilon $epsilon"
            ;;
        *)
            echo -e "${RED}Error: Unknown method '$method'. Skipping.${NC}"
            return 1
            ;;
    esac
    
    # Add optional enhancements (based on paper mentions)
    cmd+=" --use_weighted_topk"  # Advanced evaluation metrics
    
    echo -e "${YELLOW}Executing: $cmd${NC}"
    
    # Execute the command
    if eval $cmd; then
        echo -e "${GREEN}✓ Experiment completed successfully.${NC}"
    else
        echo -e "${RED}✗ Experiment FAILED. Continuing with next experiment...${NC}"
        # Don't exit, continue with other experiments
    fi
}

# --- 📝 Usage Information --- #
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --dry-run          Show experiment plan without executing"
    echo ""
    echo "This script runs comprehensive experiments for the TPaNeg method"
    echo "as described in the WACV paper: 'Cyclic Compatibility Transformation"
    echo "and Target-Prediction-aware Hard Negative Mining'"
    echo ""
    echo "Key features:"
    echo "  - Tests both IQON3000 and DeepFurniture datasets"
    echo "  - Ablation study: baseline, cycle only, TPaNeg only, full method"
    echo "  - TPaNeg curriculum learning with T_gamma scheduling"
    echo "  - PaNeg epsilon margin parameter variations"
    echo "  - Multiple seeds for statistical significance"
    echo "  - Architecture variations for robustness testing"
}

# --- 🔧 Command Line Argument Handling --- #
case "${1:-}" in
    --help|-h)
        usage
        exit 0
        ;;
    --dry-run)
        echo -e "${YELLOW}DRY RUN MODE - No experiments will be executed${NC}"
        # You could add dry-run logic here
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac