#!/bin/bash -l
#SBATCH --job-name=GHN
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="Testing Job"
#SBATCH --mail-user=slack:@ak3748       # Slack username to notify
#SBATCH --mail-type=END
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
JOB_TAG="GHN_${JOB_ID}"
OUT_FILE="$LOG_DIR/${JOB_TAG}.out"
ERR_FILE="$LOG_DIR/${JOB_TAG}.err"
LOG_FILE="$LOG_DIR/${JOB_TAG}.log"


# Pipe script stdout/stderr to files and also to console
exec > >(tee -a "$OUT_FILE" | tee -a "$LOG_FILE")
exec 2> >(tee -a "$ERR_FILE" | tee -a "$LOG_FILE" >&2)

echo "Starting GHN"
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

python lmghn3/train_ghn_ddp.py --heads 2 --vocab_size 50257 --seq_len 64 --interm_epoch 2 --epochs 12 --opt adamw --lr 4e-4 --wd 1e-2 -b 8 --amp -m 8 --name ghn3lm --hid 256 --scheduler cosine-warmup

echo "Job finished at $(date)"
