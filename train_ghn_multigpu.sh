#!/bin/bash -l
#SBATCH --job-name=GHN-I-MultiGPU-Final-16-2
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="GHN-3 Language Model Training with Multiple GPUs (DDP)"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:a100:2               # Request 2 GPUs (adjust as needed)
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # More CPUs for multiple GPUs (4 per GPU)
#SBATCH --mem=80g                      # More memory for multiple GPUs (40GB per GPU)

# =============================================================================
# Configuration - Adjust these values as needed
# =============================================================================

# Number of GPUs to use (should match --gres=gpu:a100:N above)
NUM_GPUS=${SLURM_GPUS:-2}  # Default to 4, or use SLURM_GPUS if set

# Training configuration
MODEL_NAME="GHN-I-MultiGPU-Final-16-2"
HEADS=2
LAYERS=3
SEQ_LEN=64
INTERM_EPOCH=5
EPOCHS=180              
BATCH_SIZE=16              # WikiText-2 batch size per GPU
META_BATCH_SIZE=2        # Total models across all GPUs (will be split evenly)
LR=0.0004
WD=0.01
OPTIMIZER="adam"
SCHEDULER="cosine"
LOG_INTERVAL=50
NUM_WORKERS=1
HID=32
HYPERNET="gatedgnn"
DECODER="conv"
MAX_SHAPE="1024,1024,1,1"  # Reduced from 1024,1024,1,1 to save memory

# Model filtering (for ~100K models)
INCLUDE_EMBEDDINGS=true  # Set to true to include embeddings (uses more memory)
MAX_D_MODEL=1024         # Maximum d_model for GPT Encoder/Mini GPT variants (~100K models)
MAX_LAYERS=20            # Maximum layers for GPT Encoder/Mini GPT variants (~100K models)

# =============================================================================
# Job Setup
# =============================================================================

# Extract job name from SLURM_JOB_NAME or use default
JOB_NAME=${SLURM_JOB_NAME:-GHN_LM_MultiGPU}
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

source venv/bin/activate

echo "=================================================================="
echo "GHN-3 Language Model Training - Multi-GPU (Distributed Training)"
echo "=================================================================="
echo "Job ID: $JOB_ID"
echo "Node: $NODE"
echo "Time: $(date)"
echo "DIR: $(dirname "$0")"
echo "Working Directory: $(pwd)"
echo "Training Script: train_ghn.py"
echo "Number of GPUs: $NUM_GPUS (using torchrun for DDP)"
echo ""
echo "Features:"
echo "  - Distributed Data Parallel (DDP)"
echo "  - TensorBoard logging"
echo "  - Experiment tracking"
echo "  - Memory-efficient configuration"
echo "=================================================================="

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || echo "nvidia-smi not available"

# Environment setup
export PYTHONPATH=$PYTHONPATH:./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Calculate effective batch sizes
# With N GPUs, meta_batch_size is split across GPUs
# Each GPU gets meta_batch_size // NUM_GPUS models
MODELS_PER_GPU=$((META_BATCH_SIZE / NUM_GPUS))
TOTAL_EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_GPUS))

echo ""
echo "Training Configuration:"
echo "  - Number of GPUs: $NUM_GPUS"
echo "  - Total meta_batch_size: $META_BATCH_SIZE (models across all GPUs)"
echo "  - Models per GPU: $MODELS_PER_GPU"
echo "  - WikiText-2 batch_size per GPU: $BATCH_SIZE"
echo "  - Total effective batch: $TOTAL_EFFECTIVE_BATCH"
echo "  - Max shape: $MAX_SHAPE"
echo "  - Include embeddings: $INCLUDE_EMBEDDINGS"
if [ -n "$MAX_D_MODEL" ]; then
    echo "  - Max d_model: $MAX_D_MODEL"
fi
if [ -n "$MAX_LAYERS" ]; then
    echo "  - Max layers: $MAX_LAYERS"
fi
echo ""

# Validate configuration
if [ $((META_BATCH_SIZE % NUM_GPUS)) -ne 0 ]; then
    echo "⚠️  WARNING: meta_batch_size ($META_BATCH_SIZE) is not divisible by NUM_GPUS ($NUM_GPUS)"
    echo "   This may cause uneven distribution. Consider adjusting META_BATCH_SIZE."
    echo ""
fi

# Build command arguments
TRAIN_ARGS=(
    --model_name "$MODEL_NAME"
    --heads $HEADS
    --layers $LAYERS
    --seq_len $SEQ_LEN
    --interm_epoch $INTERM_EPOCH
    --epochs $EPOCHS
    --batch_size $BATCH_SIZE
    --meta_batch_size $META_BATCH_SIZE
    --lr $LR
    --wd $WD
    --opt $OPTIMIZER
    --scheduler "$SCHEDULER"
    --log_interval $LOG_INTERVAL
    --num_workers $NUM_WORKERS
    --hid $HID
    --hypernet $HYPERNET
    --decoder $DECODER
    --max_shape "$MAX_SHAPE"
)

# Add optional flags

if [ "$INCLUDE_EMBEDDINGS" = true ]; then
    TRAIN_ARGS+=(--include_embeddings)
fi

# Add GPT Encoder/Mini GPT filtering arguments
if [ -n "$MAX_D_MODEL" ]; then
    TRAIN_ARGS+=(--max_d_model "$MAX_D_MODEL")
fi

if [ -n "$MAX_LAYERS" ]; then
    TRAIN_ARGS+=(--max_layers "$MAX_LAYERS")
fi

# Run GHN-3 training with torchrun for multi-GPU DDP
# torchrun automatically sets up DDP with environment variables
echo "Starting training with torchrun..."
echo "Command: torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS train_ghn.py ${TRAIN_ARGS[*]}"
echo ""

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train_ghn.py \
    "${TRAIN_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "=================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "   Check logs in: $LOG_DIR/${JOB_TAG}.err"
fi
echo "Job finished at $(date)"
echo "=================================================================="

exit $EXIT_CODE

