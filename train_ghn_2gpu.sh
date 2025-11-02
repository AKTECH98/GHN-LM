#!/bin/bash -l
#SBATCH --job-name=GHN-LM-training-2GPU
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="GHN-3 Language Model Training with 2 GPUs (DDP)"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:a100:2               # Request 2 GPUs
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # More CPUs for 2 GPUs (4 per GPU)
#SBATCH --mem=64g                       # More memory for 2 GPUs (32GB per GPU)

# Extract job name from SLURM_JOB_NAME or use default
JOB_NAME=${SLURM_JOB_NAME:-GHN_LM_2GPU}
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

echo "================================================================"
echo "GHN-3 Language Model Training - 2 GPUs (Distributed Training)"
echo "================================================================"
echo "Job ID: $JOB_ID"
echo "Node: $NODE"
echo "Time: $(date)"
echo "DIR: $(dirname "$0")"
echo "Working Directory: $(pwd)"
echo "Training Script: train_ghn.py"
echo "GPUs: 2 (using torchrun for DDP)"
echo "Features:"
echo "  - Distributed Data Parallel (DDP)"
echo "  - TensorBoard logging"
echo "  - Experiment tracking"
echo "  - Large OSS models excluded (memory efficient)"
echo "================================================================"

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || echo "nvidia-smi not available"

# Environment setup
export PYTHONPATH=$PYTHONPATH:./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Number of GPUs
NUM_GPUS=2

# Calculate effective batch sizes
# With 2 GPUs, meta_batch_size is split across GPUs
# Each GPU gets meta_batch_size // 2 models
META_BATCH_SIZE=8   # Total across all GPUs (4 per GPU)
BATCH_SIZE=4        # WikiText-2 batch size per GPU

echo ""
echo "Training Configuration:"
echo "  - Number of GPUs: $NUM_GPUS"
echo "  - Total meta_batch_size: $META_BATCH_SIZE (models across all GPUs)"
echo "  - Models per GPU: $((META_BATCH_SIZE / NUM_GPUS))"
echo "  - WikiText-2 batch_size per GPU: $BATCH_SIZE"
echo "  - Total effective batch: $((BATCH_SIZE * NUM_GPUS))"
echo ""

# Run GHN-3 training with torchrun for multi-GPU DDP
# torchrun automatically sets up DDP with environment variables
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train_ghn.py \
    --model_name "$JOB_ID" \
    --heads 2 \
    --layers 3 \
    --seq_len 64 \
    --interm_epoch 5 \
    --epochs 75 \
    --batch_size $BATCH_SIZE \
    --meta_batch_size $META_BATCH_SIZE \
    --lr 0.0004 \
    --wd 0.01 \
    --opt adam \
    --amp \
    --exclude_large_oss \
    --log_interval 10 \
    --hid 32 \
    --hypernet gatedgnn \
    --decoder conv \
    --max_shape "1024,1024,1,1"

EXIT_CODE=$?

echo ""
echo "================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
fi
echo "Job finished at $(date)"
echo "================================================================"

exit $EXIT_CODE

