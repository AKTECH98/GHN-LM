#!/bin/bash
# Script to run all configs through different initialization methods (excluding GHN)
# This will submit SLURM jobs for each config with each init method

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Config directory
CONFIG_DIR="LM/configs"

# Experiment base name (change this to customize experiment names)
EXPERIMENT_BASE_NAME="Benchmark_16_"

# Find all benchmark config files
CONFIG_FILES=($(ls -1 "$CONFIG_DIR"/benchmark_*.yaml 2>/dev/null | sort))

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    echo "❌ Error: No benchmark config files found in $CONFIG_DIR"
    exit 1
fi

echo "=========================================="
echo "Running All Configs with All Init Methods"
echo "=========================================="
echo "Found ${#CONFIG_FILES[@]} config files"
echo "Init methods: default"
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

# Function to generate experiment name
# Uses EXPERIMENT_BASE_NAME variable (default: "Benchmark")
# e.g., "1_tiny" -> "${EXPERIMENT_BASE_NAME}_1_tiny"
get_experiment_name() {
    local config_name="$1"
    echo "${EXPERIMENT_BASE_NAME}_${config_name}"
}

# Function to submit a job
submit_job() {
    local config_file="$1"
    local init_method="$2"
    local experiment_name="$3"
    
    # Create a temporary script for this specific job
    local temp_script=$(mktemp)
    
    # Base script template (similar to train_lm.sh)
    cat > "$temp_script" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${experiment_name}
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="Language Model Training with ${init_method^} Initialization"
#SBATCH --mail-user=slack:@ak3748
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g

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

echo "Starting Language Model Training with ${init_method^} Initialization"
echo "=============================================================="
echo "Job ID: \$JOB_ID"
echo "Node: \$NODE"
echo "Time: \$(date)"
echo "Config: ${config_file}"
echo "Init Method: ${init_method}"
echo "Experiment Name: ${experiment_name}"
echo "=============================================================="

export PYTHONPATH=\$PYTHONPATH:./
source venv/bin/activate

echo "Virtual environment activated"

# Configuration file
CONFIG_FILE="${config_file}"
INIT_METHOD="${init_method}"

# Check if config file exists
if [ -f "\$CONFIG_FILE" ]; then
    echo "Using config file: \$CONFIG_FILE"
    echo "Initialization method: ${init_method^}"
    echo "=============================================================="
    
    python train_lm.py --config "\$CONFIG_FILE" --init_method "\$INIT_METHOD"
    
else
    echo "❌ Error: Config file not found: \$CONFIG_FILE"
    exit 1
fi

echo "Job finished at \$(date)"
EOF

    # Submit the job
    local job_id=$(sbatch "$temp_script" | awk '{print $4}')
    echo "  ✅ Submitted job $job_id: $experiment_name (${init_method})"
    
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
    
    # Submit job with default init method
    experiment_name=$(get_experiment_name "$config_name")
    submit_job "$config_file" "default" "$experiment_name"
    SUBMITTED_JOBS+=("$experiment_name")
    
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total jobs submitted: ${#SUBMITTED_JOBS[@]}"
echo ""
echo "Submitted jobs:"
for job_name in "${SUBMITTED_JOBS[@]}"; do
    echo "  - $job_name"
done
echo ""
echo "Check job status with: squeue -u \$USER"
echo "=========================================="

