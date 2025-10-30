#!/bin/bash -l
#SBATCH --job-name=Evaluate_Benchmark_10
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="LM Evaluation"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g

# Extract job name from SLURM_JOB_NAME or use default
JOB_NAME=${SLURM_JOB_NAME}
# Create Job ID from job name and date
JOB_ID="${JOB_NAME}_$(date +%s)"
NODE=${SLURMD_NODENAME:-$(hostname)}

# Export JOB_ID for use by the training script
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

echo "Starting Language Model Evaluation"
echo "==================================="
echo "Job ID: $JOB_ID"
echo "Node: $NODE"
echo "Time: $(date)"
echo "DIR: $(dirname "$0")"
echo "Working Directory: $(pwd)"
echo "==================================="

# Load CUDA module if available
# module load cuda/11.7

# ===========================================
# EVALUATION PARAMETERS - MODIFY AS NEEDED
# ===========================================

# Configuration file to use
CONFIG="benchmark_10_mini_gpt_xl"  # Change to desired config

echo "Checking required files..."

# Check if required files exist
if [ ! -f "evaluate_checkpoints.py" ]; then
    echo "❌ Error: evaluate_checkpoints.py not found!"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found!"
    exit 1
fi

if [ ! -d "Experiment" ]; then
    echo "❌ Error: Experiment directory not found!"
    exit 1
fi

echo "✅ All required files present"

export PYTHONPATH=$PYTHONPATH:./
source venv/bin/activate

echo "✅ Virtual environment activated"

# Optional parameters
EPOCHS=""  # Leave empty for all epochs, or specify like "2,5,10"
# OUTPUT_FILE=""  # Leave empty to auto-save in Experiment/results_<config>/ folder

# ===========================================
# EVALUATION COMMAND
# ===========================================

echo "Starting evaluation..."
echo "Config: $CONFIG"
echo "Results will be saved in: Experiment/results_${CONFIG}/"

# Build command
CMD="python evaluate_checkpoints.py --config \"$CONFIG\" --compare"

# Add epochs if specified
if [ ! -z "$EPOCHS" ]; then
    CMD="$CMD --epochs \"$EPOCHS\""
    echo "Epochs: $EPOCHS"
fi

echo "Command: $CMD"
echo "==================================="

# Run the evaluation
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
    echo "Results saved in: Experiment/results_${CONFIG}/"
else
    echo "❌ Evaluation failed with exit code $?"
    exit 1
fi

echo "Job finished at $(date)"
