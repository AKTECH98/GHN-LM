#!/bin/bash -l
#SBATCH --job-name=GHN_D_init_10_mini_gpt_xl
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="GHN-3 Language Model Training with GHN Initialization"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g

JOB_NAME=${SLURM_JOB_NAME}

JOB_ID="${JOB_NAME}_$(date +%s)"
export JOB_ID
NODE=${SLURMD_NODENAME:-$(hostname)}

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

source venv/bin/activate

echo "Starting GHN-3 Language Model Training with GHN Initialization"
echo "=============================================================="
echo "Job ID: $JOB_ID"
echo "Node: $NODE"
echo "Time: $(date)"
echo "DIR: $(dirname "$0")"
echo "Working Directory: $(pwd)"
echo "Training Script: train_lm_ghn_init.py"
echo "Features: GHN initialization, TensorBoard logging, Experiment tracking"
echo "=============================================================="

# module load cuda/11.7

echo "All required files present"

export PYTHONPATH=$PYTHONPATH:./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration
CONFIG="LM/configs/benchmark_10_mini_gpt_xl.yaml"  # Change this to your desired config
GHN_CHECKPOINT="Experiment/GHN-D-MultiGPU/best_model.pt"  # Change this to your GHN checkpoint

echo "Configuration: $CONFIG"
echo "GHN Checkpoint: $GHN_CHECKPOINT"

# Run GHN-initialized training
python train_lm_ghn_init.py \
    --config "$CONFIG" \
    --ghn_checkpoint "$GHN_CHECKPOINT" \
    --save_ghn_init

echo "Job finished at $(date)"
