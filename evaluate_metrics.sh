#!/bin/bash -l
#SBATCH --job-name=Eval_1_tiny
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="Language Model Metrics Evaluation"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g

# Extract job name from SLURM_JOB_NAME or use default
JOB_NAME=${SLURM_JOB_NAME}
# Create Job ID from job name and date
JOB_ID="${JOB_NAME}_$(date +%s)"
NODE=${SLURMD_NODENAME:-$(hostname)}

# Export JOB_ID for use by the evaluation script
export JOB_ID

# --- logging setup ---
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
JOB_TAG="${JOB_ID}"
OUT_FILE="$LOG_DIR/${JOB_TAG}.out"
ERR_FILE="$LOG_DIR/${JOB_TAG}.err"
LOG_FILE="$LOG_DIR/${JOB_TAG}.log"

# Pipe script stdout/stderr to files and also to console
exec > >(tee -a "$OUT_FILE" | tee -a "$LOG_FILE")
exec 2> >(tee -a "$ERR_FILE" | tee -a "$LOG_FILE" >&2)

echo "Starting Language Model Metrics Evaluation"
echo "=========================================="
echo "Job ID: $JOB_ID"
echo "Node: $NODE"
echo "Time: $(date)"
echo "DIR: $(dirname "$0")"
echo "Working Directory: $(pwd)"
echo "Evaluation Script: evaluate_metrics.py"
echo "=========================================="

# module load cuda/11.7

echo "All required files present"

export PYTHONPATH=$PYTHONPATH:./
source venv/bin/activate

echo "Virtual environment activated"

# ===========================================
# EVALUATION PARAMETERS - MODIFY AS NEEDED
# ===========================================

# Configuration file to use
CONFIG_FILE=${CONFIG_FILE:-"LM/configs/benchmark_1_tiny.yaml"}  # Change to desired config

# Intervals for perplexity extraction (comma-separated epochs)
INTERVALS=${INTERVALS:-"1,2,5,10,20,50"}

# Device to use for evaluation
DEVICE=${DEVICE:-"cuda"}

# ===========================================
# EVALUATION COMMAND
# ===========================================

# Check if config file exists
if [ -f "$CONFIG_FILE" ]; then
    echo "Using config file: $CONFIG_FILE"
    echo "Intervals: $INTERVALS"
    echo "Device: $DEVICE"
    echo "=========================================="
    
    python evaluate_metrics.py \
        --config "$CONFIG_FILE" \
        --intervals "$INTERVALS" \
        --device "$DEVICE"
    
else
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    echo "Available config files:"
    ls -la LM/configs/benchmark_*.yaml 2>/dev/null || echo "  No config files found in LM/configs/"
    echo ""
    echo "Please set CONFIG_FILE to a valid config file path"
    exit 1
fi

echo "Job finished at $(date)"

