#!/bin/bash -l
#SBATCH --job-name=Benchmark_5_mini_gpt
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="Language Model Training"
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

echo "Starting Language Model Training"
echo "================================"
echo "Job ID: $JOB_ID"
echo "Node: $NODE"
echo "Time: $(date)"
echo "DIR:$(dirname "$0")"
echo "Working Directory: $(pwd)"
echo "================================"

# module load cuda/11.7

echo "All required files present"

export PYTHONPATH=$PYTHONPATH:./
source venv/bin/activate

echo "Virtual environment activated"
# ===========================================
# TRAINING PARAMETERS - MODIFY AS NEEDED
# ===========================================

# Configuration file to use
CONFIG_FILE="LM/configs/benchmark_5_mini_gpt.yaml"  # Change to desired config

# ===========================================
# TRAINING COMMAND
# ===========================================

# Check if config file exists
if [ -f "$CONFIG_FILE" ]; then
    echo "Using config file: $CONFIG_FILE"
    echo "================================"
    
    python train_lm.py --config "$CONFIG_FILE"
    
else
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    echo "Available config files:"
    ls -la LM/configs/benchmark_*.yaml 2>/dev/null || echo "  No config files found in LM/configs/"
    echo ""
    echo "Please set CONFIG_FILE to a valid config file path"
    exit 1
fi

echo "Job finished at $(date)"
