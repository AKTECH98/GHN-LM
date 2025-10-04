#!/bin/bash -l
#SBATCH --job-name=lstm
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="Language Model Training"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --mail-type=END
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4g

JOB_ID=${SLURM_JOB_ID:-local-$(date +%s)}
NODE=${SLURMD_NODENAME:-$(hostname)}

# --- logging setup ---
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
JOB_TAG="LM_${JOB_ID}"
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

# ===========================================
# TRAINING PARAMETERS - MODIFY AS NEEDED
# ===========================================
MODEL="lstm"                    # Options: rnn, lstm, gru, gpt_encoder, mini_gpt
EPOCHS=20                      # Number of training epochs
BATCH_SIZE=64                   # Batch size
D_MODEL=128                     # Model dimension
N_LAYER=2                      # Number of layers
N_HEAD=8                       # Number of attention heads (for transformers)
D_FF=2048                      # Feed-forward dimension (for transformers)
SEQ_LEN=128                     # Sequence length
LEARNING_RATE=0.001           # Learning rate
DEVICE="cuda"                  # Device: cuda or cpu

# ===========================================
# TRAINING COMMAND
# ===========================================
echo "Training Parameters:"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  D Model: $D_MODEL"
echo "  N Layer: $N_LAYER"
echo "  N Head: $N_HEAD"
echo "  D FF: $D_FF"
echo "  Seq Len: $SEQ_LEN"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Device: $DEVICE"
echo "================================"

python train_single_model.py \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --d_model $D_MODEL \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --d_ff $D_FF \
    --seq_len $SEQ_LEN \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --cache_dir "/tmp/wikitext_fixed_$(date +%s)"

echo "Job finished at $(date)"
