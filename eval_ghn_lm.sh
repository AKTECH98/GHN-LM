#!/bin/bash -l
#SBATCH --job-name=GHN-LM-Eval
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="GHN-3 Language Model Evaluation"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --mail-type=END
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g

JOB_ID=${SLURM_JOB_ID:-local-$(date +%s)}
NODE=${SLURMD_NODENAME:-$(hostname)}

# --- logging setup ---
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
JOB_TAG="GHN_EVAL_${JOB_ID}"
OUT_FILE="$LOG_DIR/${JOB_TAG}.out"
ERR_FILE="$LOG_DIR/${JOB_TAG}.err"
LOG_FILE="$LOG_DIR/${JOB_TAG}.log"

# Pipe script stdout/stderr to files and also to console
exec > >(tee -a "$OUT_FILE" | tee -a "$LOG_FILE")
exec 2> >(tee -a "$ERR_FILE" | tee -a "$LOG_FILE" >&2)

echo "Starting GHN-3 Language Model Evaluation"
echo "========================================"
echo "Job ID: $JOB_ID"
echo "Node: $NODE"
echo "Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# --- Environment Setup ---
echo "Setting up environment..."
source venv/bin/activate
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# --- Default Parameters ---
CHECKPOINT="Experiment/20917896/best_model.pt"
CONFIG="Experiment/20917896/config.json"
DEVICE=""  # Auto-detect if not specified
NUM_MODELS=50
MAX_WIKITEXT_SAMPLES=1000
OUTPUT_FILE="eval_results_${JOB_ID}.json"
VERBOSE=""

# --- Parse Command Line Arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --cuda)
            DEVICE="cuda"
            shift
            ;;
        --num_models)
            NUM_MODELS="$2"
            shift 2
            ;;
        --max_wikitext_samples)
            MAX_WIKITEXT_SAMPLES="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --ckpt PATH              Path to trained GHN checkpoint"
            echo "  --config PATH            Path to model configuration JSON file"
            echo "  --device DEVICE          Device to run evaluation on (cuda/cpu/auto)"
            echo "  --cpu                    Force CPU usage"
            echo "  --cuda                   Force CUDA usage"
            echo "  --num_models N           Number of models to evaluate (-1 for all)"
            echo "  --max_wikitext_samples N Maximum samples for WikiText-2 evaluation"
            echo "  --output_file PATH       Output file for results"
            echo "  --verbose                Verbose output"
            echo "  --help                   Show this help message"
            echo ""
            echo "Default values:"
            echo "  CHECKPOINT: $CHECKPOINT"
            echo "  CONFIG: $CONFIG"
            echo "  DEVICE: $DEVICE"
            echo "  NUM_MODELS: $NUM_MODELS"
            echo "  MAX_WIKITEXT_SAMPLES: $MAX_WIKITEXT_SAMPLES"
            echo "  OUTPUT_FILE: $OUTPUT_FILE"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# --- Validation ---
echo "Validating inputs..."
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
if [[ -n "$DEVICE" ]]; then
    echo "Device: $DEVICE (user specified)"
else
    echo "Device: auto-detect"
fi
echo "Number of models: $NUM_MODELS"
echo "Max WikiText samples: $MAX_WIKITEXT_SAMPLES"
echo "Output file: $OUTPUT_FILE"
echo ""

# --- Run Evaluation ---
echo "Starting evaluation..."
echo "======================"

# Build command with device parameter only if specified
DEVICE_ARGS=""
if [[ -n "$DEVICE" ]]; then
    DEVICE_ARGS="--device $DEVICE"
fi

python eval_ghn_lm.py \
    --ckpt "$CHECKPOINT" \
    --config "$CONFIG" \
    $DEVICE_ARGS \
    --num_models "$NUM_MODELS" \
    --max_wikitext_samples "$MAX_WIKITEXT_SAMPLES" \
    --output_file "$OUTPUT_FILE" \
    $VERBOSE

EVAL_EXIT_CODE=$?

echo ""
echo "======================"
echo "Evaluation completed with exit code: $EVAL_EXIT_CODE"

if [[ $EVAL_EXIT_CODE -eq 0 ]]; then
    echo "✅ Evaluation successful!"
    echo "Results saved to: $OUTPUT_FILE"
    
    # Display summary if results file exists
    if [[ -f "$OUTPUT_FILE" ]]; then
        echo ""
        echo "📊 Evaluation Summary:"
        echo "====================="
        python -c "
import json
try:
    with open('$OUTPUT_FILE', 'r') as f:
        results = json.load(f)
    
    print(f'Timestamp: {results.get(\"timestamp\", \"N/A\")}')
    print(f'Models evaluated: {results.get(\"models_evaluated\", \"N/A\")}')
    print(f'Total time: {results.get(\"total_time\", \"N/A\"):.2f} seconds')
    print(f'Avg time per model: {results.get(\"avg_time_per_model\", \"N/A\"):.2f} seconds')
    print(f'Avg parameters: {results.get(\"avg_parameters\", \"N/A\"):.2f}M')
    print(f'Avg parameter norm: {results.get(\"avg_param_norm\", \"N/A\"):.2f}')
    
    if results.get('avg_perplexity') is not None:
        print(f'Avg perplexity: {results.get(\"avg_perplexity\", \"N/A\"):.2f}')
        print(f'Perplexity std: {results.get(\"std_perplexity\", \"N/A\"):.2f}')
        print(f'Successful evaluations: {results.get(\"successful_evaluations\", \"N/A\")}')
    else:
        print('No successful WikiText-2 evaluations')
        
except Exception as e:
    print(f'Error reading results: {e}')
"
    fi
else
    echo "❌ Evaluation failed!"
    echo "Check the log files for details:"
    echo "  - Output: $OUT_FILE"
    echo "  - Errors: $ERR_FILE"
fi

echo ""
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"

exit $EVAL_EXIT_CODE
