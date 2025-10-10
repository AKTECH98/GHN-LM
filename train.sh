#!/bin/bash -l
#SBATCH --job-name=ghn3-lm
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="GHN-3 Language Model Training"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --mail-type=END
#SBATCH --gres=gpu:a100:2
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g

# ===========================================
# GHN-3 Language Model Training Script
# ===========================================
# This script trains a Graph HyperNetwork (GHN-3) on language models
# using WikiText-2 dataset. The GHN learns to predict parameters for
# various language model architectures (RNN, LSTM, GRU, GPT, MiniGPT).
#
# Features:
# - Comprehensive logging (TensorBoard + JSON metadata)
# - Automatic checkpoint saving (every 5 epochs + best model)
# - Validation monitoring with perplexity tracking
# - Mixed precision training support
# - Configurable GHN architecture
# - Resume from checkpoint capability
#
# Usage:
#   sbatch train.sh                    # Submit to SLURM
#   bash train.sh                      # Run locally
# ===========================================

JOB_ID=${SLURM_JOB_ID:-local-$(date +%s)}
NODE=${SLURMD_NODENAME:-$(hostname)}

# --- logging setup ---
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
JOB_TAG="GHN_${JOB_ID}"
OUT_FILE="$LOG_DIR/${JOB_TAG}.out"
ERR_FILE="$LOG_DIR/${JOB_TAG}.err"
LOG_FILE="$LOG_DIR/${JOB_TAG}.log"


# Pipe script stdout/stderr to files and also to console
exec > >(tee -a "$OUT_FILE" | tee -a "$LOG_FILE")
exec 2> >(tee -a "$ERR_FILE" | tee -a "$LOG_FILE" >&2)

echo "Starting GHN-3 Language Model Training"
echo "======================================"
echo "Job ID: $JOB_ID"
echo "Node: $NODE"
echo "Time: $(date)"
echo "DIR:$(dirname "$0")"
echo "Working Directory: $(pwd)"
echo "======================================"

# module load cuda/11.7

echo "All required files present"

export PYTHONPATH=$PYTHONPATH:./

# ===========================================
# GHN-3 TRAINING PARAMETERS - MODIFY AS NEEDED
# ===========================================
EXPERIMENT_NAME="ghn3-lm"      # Experiment name
EPOCHS=10                      # Number of training epochs
BATCH_SIZE=2                  # Batch size for language models
META_BATCH_SIZE=2              # Number of models per meta-batch
LEARNING_RATE=1e-4             # Learning rate
WEIGHT_DECAY=1e-2              # Weight decay
GRAD_CLIP=1.0                  # Gradient clipping
GHN_HID=8                     # GHN hidden dimension
GHN_LAYERS=1                   # Number of GHN layers
GHN_HEADS=2                    # Number of attention heads in GHN
MAX_SEQ_LEN=32                # Maximum sequence length
VOCAB_SIZE=1000               # Vocabulary size (auto-detected from tokenizer)
USE_ALL_CONFIGS=false          # Use full dataset (3M+ configs) or reasonable (~17K)
HYPERNET="gatedgnn"            # Hypernetwork type: gatedgnn, gnn
DECODER="conv"                 # Decoder type: conv, mlp
WEIGHT_NORM=false              # Use weight normalization
VE=false                       # Use virtual edges
LAYERNORM=false                # Use layer normalization
AMP=true                       # Use automatic mixed precision
NUM_WORKERS=2                  # Number of data loading workers
LOG_INTERVAL=100               # Logging interval
VERBOSE=false                  # Verbose logging
# CHECKPOINT_RESUME=""         # Path to checkpoint to resume from (uncomment to use)

# ===========================================
# TRAINING COMMAND
# ===========================================
echo "GHN-3 Training Parameters:"
echo "  Experiment Name: $EXPERIMENT_NAME"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Meta Batch Size: $META_BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Gradient Clip: $GRAD_CLIP"
echo "  GHN Hidden: $GHN_HID"
echo "  GHN Layers: $GHN_LAYERS"
echo "  GHN Heads: $GHN_HEADS"
echo "  Max Seq Len: $MAX_SEQ_LEN"
echo "  Vocab Size: $VOCAB_SIZE"
echo "  Use All Configs: $USE_ALL_CONFIGS"
echo "  Hypernet: $HYPERNET"
echo "  Decoder: $DECODER"
echo "  Weight Norm: $WEIGHT_NORM"
echo "  Virtual Edges: $VE"
echo "  Layer Norm: $LAYERNORM"
echo "  Mixed Precision: $AMP"
echo "  Num Workers: $NUM_WORKERS"
echo "  Log Interval: $LOG_INTERVAL"
echo "  Verbose: $VERBOSE"
echo "======================================"

# Build the training command
TRAIN_CMD="python lmghn3/train_ghn_lm.py \
    --name $EXPERIMENT_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --meta_batch_size $META_BATCH_SIZE \
    --lr $LEARNING_RATE \
    --wd $WEIGHT_DECAY \
    --grad_clip $GRAD_CLIP \
    --hid $GHN_HID \
    --layers $GHN_LAYERS \
    --heads $GHN_HEADS \
    --max_seq_len $MAX_SEQ_LEN \
    --vocab_size $VOCAB_SIZE \
    --hypernet $HYPERNET \
    --decoder $DECODER \
    --num_workers $NUM_WORKERS \
    --log_interval $LOG_INTERVAL \
    --save ./checkpoints"

# Add optional flags
if [ "$USE_ALL_CONFIGS" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_all_configs"
fi

if [ "$WEIGHT_NORM" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --weight_norm"
fi

if [ "$VE" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --ve"
fi

if [ "$LAYERNORM" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --layernorm"
fi

if [ "$AMP" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --amp"
fi

if [ "$VERBOSE" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --verbose"
fi

# Add checkpoint resumption if specified
if [ ! -z "$CHECKPOINT_RESUME" ]; then
    TRAIN_CMD="$TRAIN_CMD --ckpt $CHECKPOINT_RESUME"
    echo "  Resuming from checkpoint: $CHECKPOINT_RESUME"
fi

# Execute the training command
echo "Executing: $TRAIN_CMD"
eval $TRAIN_CMD

echo "Job finished at $(date)"
