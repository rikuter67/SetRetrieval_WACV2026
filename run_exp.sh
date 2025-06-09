#!/bin/bash
# WACV 2025 Experiment Runner - Factorial Design

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
GPU=0
DATASET="DeepFurniture"
EXPERIMENT_TYPE="all"
MODE="train"

# Experiment dimensions (factorial design)
ARCHITECTURES="4L4H"           # Default: 4L4H
METHODS="baseline"             # Default: baseline only
BATCH_SIZES="128"              # Default: 128
CYCLE_LAMBDAS="0.2"           # Default: 0.2
NEG_NUMS="10"                 # Default: 10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --arch) ARCHITECTURES="$2"; shift 2 ;;
        --methods) METHODS="$2"; shift 2 ;;
        --batch) BATCH_SIZES="$2"; shift 2 ;;
        --lambdas) CYCLE_LAMBDAS="$2"; shift 2 ;;
        --negs) NEG_NUMS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [EXPERIMENT_TYPE]"
            echo ""
            echo "Basic Options:"
            echo "  --gpu NUM            GPU to use (0 or 1)"
            echo "  --dataset NAME       DeepFurniture or IQON3000 (default: DeepFurniture)"
            echo ""
            echo "Factorial Design Options (comma-separated lists):"
            echo "  --arch LIST          Architectures: 2L2H,4L4H,6L6H"
            echo "  --methods LIST       Methods: baseline,cycle,clneg,full"
            echo "  --batch LIST         Batch sizes: 64,128 (default: 64,128)"
            echo "  --lambdas LIST       Cycle lambda values: 0.1,0.2,0.5"
            echo "  --negs LIST          CLNeg neg_num values: 10,20,30"
            echo ""
            echo "Experiment Types:"
            echo "  custom              Use specified parameters (factorial combination)"
            echo "  ablation            Method comparison: baseline,cycle,clneg,full"
            echo "  architecture        Architecture comparison: 2L2H,4L4H,6L6H"
            echo "  hyperparameter      Batch size comparison: 64,128"
            echo "  cycle-sweep         Lambda sweep: 0.1,0.2,0.5"
            echo "  clneg-sweep         Neg_num sweep: 10,20,30"
            echo "  all                 Complete factorial design"
            echo ""
            echo "Examples:"
            echo "  $0 --gpu 0 ablation                    # Compare 4 methods (4L4H, batch=128)"
            echo "  $0 --gpu 1 architecture                # Compare 3 architectures (full method)"
            echo "  $0 --gpu 0 --arch 4L4H,6L6H --methods baseline,full custom"
            echo "  $0 --gpu 1 all                         # Full factorial: 3×4×2×3×3 = 216 experiments"
            echo ""
            echo "Method Definitions:"
            echo "  baseline  = center-base only"
            echo "  cycle     = center-base + cycle consistency"
            echo "  clneg     = center-base + CLNeg"
            echo "  full      = center-base + cycle + CLNeg"
            exit 0
            ;;
        *) EXPERIMENT_TYPE="$1"; shift ;;
    esac
done

# Set experiment parameters based on type
case "$EXPERIMENT_TYPE" in
    "ablation")
        ARCHITECTURES="4L4H"
        METHODS="baseline,cycle,clneg,full"
        BATCH_SIZES="128"
        CYCLE_LAMBDAS="0.2"
        NEG_NUMS="10"
        ;;
    "architecture")
        ARCHITECTURES="2L2H,4L4H,6L6H"
        METHODS="full"
        BATCH_SIZES="128"
        CYCLE_LAMBDAS="0.2"
        NEG_NUMS="10"
        ;;
    "hyperparameter")
        ARCHITECTURES="4L4H"
        METHODS="full"
        BATCH_SIZES="128"
        CYCLE_LAMBDAS="0.2"
        NEG_NUMS="10"
        ;;
    "cycle-sweep")
        ARCHITECTURES="4L4H"
        METHODS="cycle,full"
        BATCH_SIZES="128"
        CYCLE_LAMBDAS="0.1,0.2,0.5"
        NEG_NUMS="10"
        ;;
    "clneg-sweep")
        ARCHITECTURES="4L4H"
        METHODS="clneg,full"
        BATCH_SIZES="128"
        CYCLE_LAMBDAS="0.2"
        NEG_NUMS="10,20,30"
        ;;
    "all")
        # Set defaults for 'all', but don't override user-specified parameters
        [[ "$ARCHITECTURES" == "4L4H" ]] && ARCHITECTURES="2L2H,4L4H,6L6H"
        [[ "$METHODS" == "baseline" ]] && METHODS="baseline,cycle,clneg,full"
        [[ "$BATCH_SIZES" == "128" ]] && BATCH_SIZES="128"
        [[ "$CYCLE_LAMBDAS" == "0.2" ]] && CYCLE_LAMBDAS="0.1,0.2,0.5"
        [[ "$NEG_NUMS" == "10" ]] && NEG_NUMS="10,20,30"
        ;;
    "custom")
        # Use user-specified parameters
        ;;
    *)
        echo "Unknown experiment type: $EXPERIMENT_TYPE"
        echo "Use --help for available options"
        exit 1
        ;;
esac

# Function to build method arguments
build_method_args() {
    local method="$1"
    local cycle_lambda="$2"
    local neg_num="$3"
    local args=""
    
    case "$method" in
        "baseline")
            args="--use-center-base"
            ;;
        "cycle")
            args="--use-center-base --use-cycle-loss --cycle-lambda $cycle_lambda"
            ;;
        "clneg")
            args="--use-center-base --use-clneg-loss --neg-num $neg_num"
            ;;
        "full")
            args="--use-center-base --use-cycle-loss --cycle-lambda $cycle_lambda --use-clneg-loss --neg-num $neg_num"
            ;;
        *)
            echo "Unknown method: $method"
            exit 1
            ;;
    esac
    
    echo "$args"
}

# Function to run single experiment
run_experiment() {
    local arch="$1"
    local method="$2"
    local batch_size="$3"
    local cycle_lambda="$4"
    local neg_num="$5"
    local exp_num="$6"
    local total_exp="$7"
    
    # Parse architecture (handle both "4L4H" and "4-4" formats)
    if [[ "$arch" =~ ^([0-9]+)L([0-9]+)H$ ]]; then
        local layers="${BASH_REMATCH[1]}"
        local heads="${BASH_REMATCH[2]}"
    else
        echo "Error: Invalid architecture format: $arch"
        echo "Expected format: XLYkH (e.g., 4L4H)"
        return 1
    fi
    
    # Build method args
    local method_args=$(build_method_args "$method" "$cycle_lambda" "$neg_num")
    
    # Data directory for IQON3000
    local data_dir=""
    if [[ "$DATASET" == "datasets/DeepFurniture" ]]; then
        data_dir="--data-dir DeepFureniture"
    fi
    
    # Create experiment name
    local exp_name="${arch}_${method}_B${batch_size}"
    if [[ "$method" == "cycle" || "$method" == "full" ]]; then
        exp_name="${exp_name}_L${cycle_lambda}"
    fi
    if [[ "$method" == "clneg" || "$method" == "full" ]]; then
        exp_name="${exp_name}_N${neg_num}"
    fi
    
    echo -e "${BLUE}[$exp_num/$total_exp] $exp_name${NC}"
    echo "  Architecture: $layers layers, $heads heads"
    echo "  Method: $method"
    echo "  Batch size: $batch_size"
    if [[ "$method" == "cycle" || "$method" == "full" ]]; then
        echo "  Cycle lambda: $cycle_lambda"
    fi
    if [[ "$method" == "clneg" || "$method" == "full" ]]; then
        echo "  Neg num: $neg_num"
    fi
    echo ""
    
    # Run experiment
    if CUDA_VISIBLE_DEVICES=$GPU python run.py \
        --dataset $DATASET \
        --mode "$MODE" \
        --batch-size $batch_size \
        --epochs 100 \
        --num-layers $layers \
        --num-heads $heads \
        --learning-rate 1e-4 \
        --patience 10 \
        $data_dir \
        $method_args; then
        echo -e "${GREEN}✓ $exp_name completed${NC}"
        return 0
    else
        echo -e "${RED}✗ $exp_name failed${NC}"
        return 1
    fi
}

# Function to calculate total experiments
calculate_total_experiments() {
    local arch_count=$(echo "$ARCHITECTURES" | tr ',' '\n' | wc -l)
    local method_count=$(echo "$METHODS" | tr ',' '\n' | wc -l)
    local batch_count=$(echo "$BATCH_SIZES" | tr ',' '\n' | wc -l)
    local lambda_count=$(echo "$CYCLE_LAMBDAS" | tr ',' '\n' | wc -l)
    local neg_count=$(echo "$NEG_NUMS" | tr ',' '\n' | wc -l)
    
    # Calculate based on methods that actually use each parameter
    local total=0
    
    for method in $(echo "$METHODS" | tr ',' ' '); do
        case "$method" in
            "baseline")
                total=$((total + arch_count * batch_count))
                ;;
            "cycle")
                total=$((total + arch_count * batch_count * lambda_count))
                ;;
            "clneg")
                total=$((total + arch_count * batch_count * neg_count))
                ;;
            "full")
                total=$((total + arch_count * batch_count * lambda_count * neg_count))
                ;;
        esac
    done
    
    echo $total
}

# Main execution
main() {
    # Calculate total experiments
    local total_experiments=$(calculate_total_experiments)
    
    echo "=================================================================="
    echo "           WACV 2025 Factorial Design Experiments"
    echo "=================================================================="
    echo "GPU: $GPU"
    echo "Dataset: $DATASET"
    echo "Experiment Type: $EXPERIMENT_TYPE"
    echo ""
    echo "Factorial Design Parameters:"
    echo "  Architectures: $ARCHITECTURES"
    echo "  Methods: $METHODS"  
    echo "  Batch sizes: $BATCH_SIZES"
    echo "  Cycle lambdas: $CYCLE_LAMBDAS"
    echo "  Neg nums: $NEG_NUMS"
    echo ""
    echo "Total experiments: $total_experiments"
    echo "Estimated time: $((total_experiments * 1))-$((total_experiments * 2)) hours"
    echo "=================================================================="
    echo ""
    
    local exp_count=0
    local start_time=$(date +%s)
    
    # Run factorial experiments
    for arch in $(echo "$ARCHITECTURES" | tr ',' ' '); do
        for method in $(echo "$METHODS" | tr ',' ' '); do
            for batch_size in $(echo "$BATCH_SIZES" | tr ',' ' '); do
                case "$method" in
                    "baseline")
                        exp_count=$((exp_count + 1))
                        run_experiment "$arch" "$method" "$batch_size" "0.2" "10" "$exp_count" "$total_experiments"
                        ;;
                    "cycle")
                        for cycle_lambda in $(echo "$CYCLE_LAMBDAS" | tr ',' ' '); do
                            exp_count=$((exp_count + 1))
                            run_experiment "$arch" "$method" "$batch_size" "$cycle_lambda" "10" "$exp_count" "$total_experiments"
                        done
                        ;;
                    "clneg")
                        for neg_num in $(echo "$NEG_NUMS" | tr ',' ' '); do
                            exp_count=$((exp_count + 1))
                            run_experiment "$arch" "$method" "$batch_size" "0.2" "$neg_num" "$exp_count" "$total_experiments"
                        done
                        ;;
                    "full")
                        for cycle_lambda in $(echo "$CYCLE_LAMBDAS" | tr ',' ' '); do
                            for neg_num in $(echo "$NEG_NUMS" | tr ',' ' '); do
                                exp_count=$((exp_count + 1))
                                run_experiment "$arch" "$method" "$batch_size" "$cycle_lambda" "$neg_num" "$exp_count" "$total_experiments" || true
                            done
                        done
                        ;;
                esac
            done
        done
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    echo "=================================================================="
    echo "All factorial experiments completed!"
    echo "Total experiments: $exp_count"
    echo "Total time: ${hours}h ${minutes}m"
    echo "=================================================================="
}

# Run main
main