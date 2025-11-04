#!/bin/bash -l
#SBATCH --job-name=GHN-LM-training
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="GHN-3 Language Model Training with TensorBoard"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g

# Extract job name from SLURM_JOB_NAME or use default
JOB_NAME=${SLURM_JOB_NAME:-GHN_LM}
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

echo "Starting GHN-3 Language Model Training"
echo "======================================"
echo "Job ID: $JOB_ID"
echo "Node: $NODE"
echo "Time: $(date)"
echo "DIR: $(dirname "$0")"
echo "Working Directory: $(pwd)"
echo "Training Script: train_ghn.py"
echo "Features: TensorBoard logging, Experiment tracking, Organized config structure"
echo "======================================"

# module load cuda/11.7

echo "All required files present"

export PYTHONPATH=$PYTHONPATH:./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run GHN-3 training with enhanced features
# Using the updated train_ghn.py script with organized config structure
python train_ghn.py \
    --model_name "GHN-3-Try" \
    --heads 2\
    --layers 3 \
    --seq_len 64 \
    --interm_epoch 1 \
    --save "Experiment/GHN-3-T" \
    --epochs 75 \
    --batch_size 2 \
    --meta_batch_size 4 \
    --lr 0.0004 \
    --wd 0.01 \
    --opt adam \
    --amp \
    --include_embeddings \
    --exclude_oss \
    --log_interval 2 \
    --hid 32 \
    --hypernet gatedgnn \
    --decoder conv \
    --max_shape "1024,1024,1,1"

echo "Job finished at $(date)"
