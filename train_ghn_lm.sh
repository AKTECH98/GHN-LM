#!/bin/bash -l
#SBATCH --job-name=GHN-LM-From-Checkpoint
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="GHN-3 Language Model Training with TensorBoard"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --mail-type=END
#SBATCH --gres=gpu:a100:2
#SBATCH --time=0-20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128g

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

echo "Starting GHN-3 Language Model Training-Change_Shape"
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

# Run GHN-3 training with enhanced features
# Using the new train_ghn_lm.py script with TensorBoard and experiment tracking
python lmghn3/train_ghn_lm.py \
    --heads 2 \
    --vocab_size 50257 \
    --seq_len 64 \
    --interm_epoch 2 \
    --epochs 12 \
    --opt adamw \
    --lr 2e-4 \
    --wd 1e-2 \
    -b 4 \
    --amp \
    -m 4 \
    --name ghn3lm_stable \
    --hid 256 \
    --scheduler cosine-warmup \
    --grad_clip 0.5

echo "Job finished at $(date)"
echo "Experiment files organized in: checkpoints/experiment_[ID]_[NAME]/"
echo "  - Best model: checkpoints/experiment_[ID]_[NAME]/best_model/"
echo "  - Periodic checkpoints: checkpoints/experiment_[ID]_[NAME]/checkpoints/"
echo "  - Metadata: checkpoints/experiment_[ID]_[NAME]/metadata/"
echo "  - TensorBoard logs: checkpoints/tensorboard_logs/"
echo "  - TensorBoard location saved in experiment metadata"
echo "To view TensorBoard: tensorboard --logdir=checkpoints/tensorboard_logs"
