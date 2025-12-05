#!/bin/bash
# Script to run all configs with GHN initialization
# This will submit SLURM jobs for each config with GHN init method
#
# Usage: ./run_all_configs_ghn_init.sh [GHN_CHECKPOINT]
#   If GHN_CHECKPOINT is not provided, it will use the GHN_CHECKPOINT environment variable
#   If neither is set, the script will exit with an error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get GHN checkpoint from argument or environment variable
if [ -n "$1" ]; then
    GHN_CHECKPOINT="$1"
elif [ -n "$GHN_CHECKPOINT" ]; then
    # Use environment variable
    :
else
    echo "❌ Error: GHN_CHECKPOINT must be provided as an argument or environment variable"
    echo "Usage: $0 [GHN_CHECKPOINT]"
    echo "   or: GHN_CHECKPOINT=/path/to/checkpoint.pt $0"
    exit 1
fi

# Validate checkpoint file exists
if [ ! -f "$GHN_CHECKPOINT" ]; then
    echo "❌ Error: GHN checkpoint file not found: $GHN_CHECKPOINT"
    exit 1
fi

# Config directory
CONFIG_DIR="LM/configs"

# Find all benchmark config files
CONFIG_FILES=($(ls -1 "$CONFIG_DIR"/benchmark_*.yaml 2>/dev/null | sort))

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    echo "❌ Error: No benchmark config files found in $CONFIG_DIR"
    exit 1
fi

echo "=========================================="
echo "Running All Configs with GHN Init"
echo "=========================================="
echo "Found ${#CONFIG_FILES[@]} config files"
echo "GHN Checkpoint: $GHN_CHECKPOINT"
echo "Total jobs to submit: ${#CONFIG_FILES[@]}"
echo "=========================================="
echo ""

# Function to extract config number and name from file path
# e.g., "LM/configs/benchmark_1_tiny.yaml" -> "1_tiny"
get_config_name() {
    local config_file="$1"
    local basename=$(basename "$config_file" .yaml)
    # Remove "benchmark_" prefix
    echo "${basename#benchmark_}"
}

# Function to generate experiment name for GHN init
# e.g., "1_tiny" -> "GHN-I_1_tiny"
get_experiment_name() {
    local config_name="$1"
    echo "GHN-I_${config_name}"
}

# Function to submit a job
submit_job() {
    local config_file="$1"
    local experiment_name="$2"
    local ghn_checkpoint="$3"
    
    # Create a temporary script for this specific job
    local temp_script=$(mktemp)
    
    # Base script template (similar to train_lm_ghn_init.sh)
    cat > "$temp_script" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${experiment_name}
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="Language Model Training with GHN Initialization"
#SBATCH --mail-user=slack:@ak3748
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g

# Extract job name from SLURM_JOB_NAME or use default
JOB_NAME=\${SLURM_JOB_NAME}
# Create Job ID from job name and date
JOB_ID="\${JOB_NAME}_\$(date +%s)"
NODE=\${SLURMD_NODENAME:-\$(hostname)}

# Export JOB_ID for use by the training script
export JOB_ID

# --- logging setup ---
LOG_DIR="logs"
mkdir -p "\$LOG_DIR"
JOB_TAG="\${JOB_ID}"
OUT_FILE="\$LOG_DIR/\${JOB_TAG}.out"
ERR_FILE="\$LOG_DIR/\${JOB_TAG}.err"
LOG_FILE="\$LOG_DIR/\${JOB_TAG}.log"

# Pipe script stdout/stderr to files and also to console
exec > >(tee -a "\$OUT_FILE" | tee -a "\$LOG_FILE")
exec 2> >(tee -a "\$ERR_FILE" | tee -a "\$LOG_FILE" >&2)

echo "Starting Language Model Training with GHN Initialization"
echo "=============================================================="
echo "Job ID: \$JOB_ID"
echo "Node: \$NODE"
echo "Time: \$(date)"
echo "Config: ${config_file}"
echo "GHN Checkpoint: ${ghn_checkpoint}"
echo "Experiment Name: ${experiment_name}"
echo "=============================================================="

export PYTHONPATH=\$PYTHONPATH:./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source venv/bin/activate

echo "Virtual environment activated"

# Configuration file
CONFIG_FILE="${config_file}"
GHN_CHECKPOINT="${ghn_checkpoint}"

# Check if config file exists
if [ -f "\$CONFIG_FILE" ]; then
    echo "Using config file: \$CONFIG_FILE"
    echo "GHN Checkpoint: \$GHN_CHECKPOINT"
    echo "Initialization method: GHN"
    echo "=============================================================="
    
    python train_lm.py \
        --config "\$CONFIG_FILE" \
        --init_method ghn \
        --ghn_checkpoint "\$GHN_CHECKPOINT" \
        --save_ghn_init
    
else
    echo "❌ Error: Config file not found: \$CONFIG_FILE"
    exit 1
fi

echo "Job finished at \$(date)"
EOF

    # Submit the job
    local job_id=$(sbatch "$temp_script" | awk '{print $4}')
    echo "  ✅ Submitted job $job_id: $experiment_name"
    
    # Clean up temp script
    rm "$temp_script"
    
    return 0
}

# Track submitted jobs
SUBMITTED_JOBS=()

# Loop through all config files
for config_file in "${CONFIG_FILES[@]}"; do
    config_name=$(get_config_name "$config_file")
    echo "Processing config: $config_name"
    
    # Submit job with GHN init
    experiment_name=$(get_experiment_name "$config_name")
    submit_job "$config_file" "$experiment_name" "$GHN_CHECKPOINT"
    SUBMITTED_JOBS+=("$experiment_name")
    
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total jobs submitted: ${#SUBMITTED_JOBS[@]}"
echo "GHN Checkpoint: $GHN_CHECKPOINT"
echo ""
echo "Submitted jobs:"
for job_name in "${SUBMITTED_JOBS[@]}"; do
    echo "  - $job_name"
done
echo ""
echo "Check job status with: squeue -u \$USER"
echo "=========================================="

