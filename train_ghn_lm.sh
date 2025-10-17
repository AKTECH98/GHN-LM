#!/bin/bash -l
#SBATCH --job-name=GHN-LM-exclude_embeddings
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="GHN-3 Language Model Training with TensorBoard"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --mail-type=END
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g

JOB_ID=${SLURM_JOB_ID:-local-$(date +%s)}
NODE=${SLURMD_NODENAME:-$(hostname)}

# --- logging setup ---
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
JOB_TAG="GHN_LM_${JOB_ID}"
OUT_FILE="$LOG_DIR/${JOB_TAG}.out"
ERR_FILE="$LOG_DIR/${JOB_TAG}.err"
LOG_FILE="$LOG_DIR/${JOB_TAG}.log"

# Pipe script stdout/stderr to files and also to console
exec > >(tee -a "$OUT_FILE" | tee -a "$LOG_FILE")
exec 2> >(tee -a "$ERR_FILE" | tee -a "$LOG_FILE" >&2)

echo "Starting GHN-3 Language Model Training-include_embeddings"
echo "======================================"
echo "Job ID: $JOB_ID"
echo "Node: $NODE"
echo "Time: $(date)"
echo "DIR: $(dirname "$0")"
echo "Working Directory: $(pwd)"
echo "Training Script: train_ghn_lm.py"
echo "Features: TensorBoard logging, Experiment tracking, Perplexity overflow fix"
echo "======================================"

# module load cuda/11.7

echo "All required files present"

export PYTHONPATH=$PYTHONPATH:./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run GHN-3 training with enhanced features
# Using the new train_ghn_lm.py script with TensorBoard and experiment tracking
python lmghn3/train_ghn_lm.py \
    --heads 2 \
    --seq_len 64 \
    --interm_epoch 1 \
    --epochs 100 \
    --amp \
    --include_embeddings \
    --log_interval 10

echo "Job finished at $(date)"
