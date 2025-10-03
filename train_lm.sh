#!/bin/bash -l
#SBATCH --job-name=LM
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="Language Model Training"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
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
MODEL="mini_gpt"                    # Options: rnn, lstm, gru, gpt_encoder, mini_gpt
EPOCHS=20                      # Number of training epochs
BATCH_SIZE=16                   # Batch size
D_MODEL=128                     # Model dimension
N_LAYER=2                      # Number of layers
N_HEAD=8                       # Number of attention heads (for transformers)
D_FF=1024                      # Feed-forward dimension (for transformers)
SEQ_LEN=128                     # Sequence length
LEARNING_RATE=0.001            # Learning rate
WEIGHT_DECAY=0.01              # Weight decay
DROPOUT=0.1                    # Dropout rate
ATTN_DROP=0.1                  # Attention dropout rate
OPTIMIZER="adamw"              # Optimizer: adam, adamw, sgd
SCHEDULER="cosine"             # Scheduler: cosine, linear, constant, step
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
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Dropout: $DROPOUT"
echo "  Attn Drop: $ATTN_DROP"
echo "  Optimizer: $OPTIMIZER"
echo "  Scheduler: $SCHEDULER"
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
    --weight_decay $WEIGHT_DECAY \
    --dropout $DROPOUT \
    --attn_drop $ATTN_DROP \
    --optimizer $OPTIMIZER \
    --scheduler $SCHEDULER \
    --device $DEVICE

echo "Job finished at $(date)"
