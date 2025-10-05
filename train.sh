#!/bin/bash -l
#SBATCH --job-name=ghn3-lm
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="GHN-3 Language Model Training"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --mail-type=END
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g

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
EPOCHS=50                      # Number of training epochs
BATCH_SIZE=4                   # Batch size for language models
META_BATCH_SIZE=8              # Number of models per meta-batch
LEARNING_RATE=1e-4             # Learning rate
WEIGHT_DECAY=1e-2              # Weight decay
GHN_HID=64                     # GHN hidden dimension
GHN_LAYERS=3                   # Number of GHN layers
GHN_HEADS=8                    # Number of attention heads in GHN
USE_ALL_CONFIGS=false          # Use full dataset (3M+ configs) or reasonable (~17K)
HYPERNET="gatedgnn"            # Hypernetwork type: gatedgnn, gnn
DECODER="conv"                 # Decoder type: conv, mlp
WEIGHT_NORM=false              # Use weight normalization
VE=false                       # Use virtual edges
LAYERNORM=false                # Use layer normalization
IS_GHN2=false                  # Use GHN-2 instead of GHN-3
DEVICE="cuda"                  # Device: cuda or cpu

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
echo "  GHN Hidden: $GHN_HID"
echo "  GHN Layers: $GHN_LAYERS"
echo "  GHN Heads: $GHN_HEADS"
echo "  Use All Configs: $USE_ALL_CONFIGS"
echo "  Hypernet: $HYPERNET"
echo "  Decoder: $DECODER"
echo "  Weight Norm: $WEIGHT_NORM"
echo "  Virtual Edges: $VE"
echo "  Layer Norm: $LAYERNORM"
echo "  Is GHN-2: $IS_GHN2"
echo "  Device: $DEVICE"
echo "======================================"

# Build the training command
TRAIN_CMD="python lmghn3/train_lm_ghn.py \
    --name $EXPERIMENT_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --meta_batch_size $META_BATCH_SIZE \
    --lr $LEARNING_RATE \
    --wd $WEIGHT_DECAY \
    --hid $GHN_HID \
    --layers $GHN_LAYERS \
    --heads $GHN_HEADS \
    --hypernet $HYPERNET \
    --decoder $DECODER \
    --device $DEVICE"

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

if [ "$IS_GHN2" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --is_ghn2"
fi

# Execute the training command
echo "Executing: $TRAIN_CMD"
eval $TRAIN_CMD

echo "Job finished at $(date)"
