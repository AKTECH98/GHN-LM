#!/bin/bash
# Script to run evaluations for the first 10 configs
# This will submit SLURM jobs for each config to evaluate metrics
#
# Usage: ./run_first_10_configs_evaluation.sh [INTERVALS] [DEVICE]
#   INTERVALS: Comma-separated epochs for perplexity extraction (default: "1,2,5,10,20,50")
#   DEVICE: Device to use for evaluation (default: "cuda")

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get optional parameters
INTERVALS=${1:-"1,2,5,10,20,50"}
DEVICE=${2:-"cuda"}

# Config directory
CONFIG_DIR="LM/configs"

# Find all benchmark config files and take only the first 10
CONFIG_FILES=($(ls -1 "$CONFIG_DIR"/benchmark_*.yaml 2>/dev/null | sort | head -n 10))

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    echo "❌ Error: No benchmark config files found in $CONFIG_DIR"
    exit 1
fi

echo "=========================================="
echo "Running Evaluations for First 10 Configs"
echo "=========================================="
echo "Found ${#CONFIG_FILES[@]} config files (first 10)"
echo "Intervals: $INTERVALS"
echo "Device: $DEVICE"
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

# Function to generate job name for evaluation
# e.g., "1_tiny" -> "Eval_1_tiny"
get_job_name() {
    local config_name="$1"
    echo "Eval_${config_name}"
}

# Function to submit a job
submit_job() {
    local config_file="$1"
    local job_name="$2"
    local intervals="$3"
    local device="$4"
    
    # Create a temporary script for this specific job
    local temp_script=$(mktemp)
    
    # Base script template (similar to evaluate_metrics.sh)
    cat > "$temp_script" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${job_name}
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="Language Model Metrics Evaluation"
#SBATCH --mail-user=slack:@ak3748
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g

# Extract job name from SLURM_JOB_NAME or use default
JOB_NAME=\${SLURM_JOB_NAME}
# Create Job ID from job name and date
JOB_ID="\${JOB_NAME}_\$(date +%s)"
NODE=\${SLURMD_NODENAME:-\$(hostname)}

# Export JOB_ID for use by the evaluation script
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

echo "Starting Language Model Metrics Evaluation"
echo "=========================================="
echo "Job ID: \$JOB_ID"
echo "Node: \$NODE"
echo "Time: \$(date)"
echo "Config: ${config_file}"
echo "Intervals: ${intervals}"
echo "Device: ${device}"
echo "=========================================="

export PYTHONPATH=\$PYTHONPATH:./
source venv/bin/activate

echo "Virtual environment activated"

# Configuration file
CONFIG_FILE="${config_file}"
INTERVALS="${intervals}"
DEVICE="${device}"

# Check if config file exists
if [ -f "\$CONFIG_FILE" ]; then
    echo "Using config file: \$CONFIG_FILE"
    echo "Intervals: \$INTERVALS"
    echo "Device: \$DEVICE"
    echo "=========================================="
    
    python evaluate_metrics.py \
        --config "\$CONFIG_FILE" \
        --intervals "\$INTERVALS" \
        --device "\$DEVICE"
    
else
    echo "❌ Error: Config file not found: \$CONFIG_FILE"
    exit 1
fi

echo "Job finished at \$(date)"
EOF

    # Submit the job
    local job_id=$(sbatch "$temp_script" | awk '{print $4}')
    echo "  ✅ Submitted job $job_id: $job_name"
    
    # Clean up temp script
    rm "$temp_script"
    
    return 0
}

# Track submitted jobs
SUBMITTED_JOBS=()

# Loop through the first 10 config files
for config_file in "${CONFIG_FILES[@]}"; do
    config_name=$(get_config_name "$config_file")
    echo "Processing config: $config_name"
    
    # Submit evaluation job
    job_name=$(get_job_name "$config_name")
    submit_job "$config_file" "$job_name" "$INTERVALS" "$DEVICE"
    SUBMITTED_JOBS+=("$job_name")
    
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total jobs submitted: ${#SUBMITTED_JOBS[@]}"
echo "Intervals: $INTERVALS"
echo "Device: $DEVICE"
echo ""
echo "Submitted jobs:"
for job_name in "${SUBMITTED_JOBS[@]}"; do
    echo "  - $job_name"
done
echo ""
echo "Check job status with: squeue -u \$USER"
echo "=========================================="

