#!/bin/bash
# Script to submit SLURM jobs for evaluation of all configs with all init methods
# This will submit individual SLURM jobs for each benchmark config
#
# Usage: 
#   ./run_all_configs_evaluation.sh                    # Submit jobs for all configs
#   ./run_all_configs_evaluation.sh --device cpu       # Use CPU instead of CUDA
#   ./run_all_configs_evaluation.sh --init_methods default,GHN-I  # Evaluate specific init methods

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default parameters
DEVICE="cuda"
INIT_METHODS="default,GHN-I,GHN-T"
OUTPUT_DIR="Evaluations"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --init_methods)
            INIT_METHODS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--device DEVICE] [--init_methods METHODS] [--output DIR]"
            exit 1
            ;;
    esac
done

# Config directory
CONFIG_DIR="LM/configs"

# Find all benchmark config files
CONFIG_FILES=($(ls -1 "$CONFIG_DIR"/benchmark_*.yaml 2>/dev/null | sort))

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    echo "❌ Error: No benchmark config files found in $CONFIG_DIR"
    exit 1
fi

echo "=========================================="
echo "Submitting Evaluation Jobs for All Configs"
echo "=========================================="
echo "Found ${#CONFIG_FILES[@]} config files"
echo "Init methods: $INIT_METHODS"
echo "Device: $DEVICE"
echo "Output directory: $OUTPUT_DIR"
echo "Total jobs to submit: ${#CONFIG_FILES[@]}"
echo "=========================================="
echo ""

# Function to extract config number from file path
# e.g., "LM/configs/benchmark_1.yaml" -> "1"
get_config_num() {
    local config_file="$1"
    local basename=$(basename "$config_file" .yaml)
    # Extract number from "benchmark_N"
    echo "${basename#benchmark_}"
}

# Function to generate job name for evaluation
# e.g., "1" -> "Eval_1"
get_job_name() {
    local config_num="$1"
    echo "Eval_${config_num}"
}

# Function to submit a job
submit_job() {
    local config_file="$1"
    local config_num="$2"
    local job_name="$3"
    local device="$4"
    local init_methods="$5"
    local output_dir="$6"
    
    # Create a temporary script for this specific job
    local temp_script=$(mktemp)
    
    # Base script template
    cat > "$temp_script" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${job_name}
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --comment="Evaluation of Benchmark Config ${config_num}"
#SBATCH --mail-user=slack:@ak3748
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g

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

echo "Starting Evaluation for Benchmark Config ${config_num}"
echo "=========================================="
echo "Job ID: \$JOB_ID"
echo "Node: \$NODE"
echo "Time: \$(date)"
echo "Config: ${config_file}"
echo "Config Num: ${config_num}"
echo "Init Methods: ${init_methods}"
echo "Device: ${device}"
echo "Output Dir: ${output_dir}"
echo "=========================================="

export PYTHONPATH=\$PYTHONPATH:./
source venv/bin/activate

echo "Virtual environment activated"

# Configuration
CONFIG_FILE="${config_file}"
INIT_METHODS="${init_methods}"
DEVICE="${device}"
OUTPUT_DIR="${output_dir}"

# Check if config file exists
if [ -f "\$CONFIG_FILE" ]; then
    echo "Using config file: \$CONFIG_FILE"
    echo "Init methods: \$INIT_METHODS"
    echo "Device: \$DEVICE"
    echo "=========================================="
    
    python evaluate_init_methods.py \\
        --config "\$CONFIG_FILE" \\
        --init_methods "\$INIT_METHODS" \\
        --device "\$DEVICE" \\
        --output_dir "\$OUTPUT_DIR"
    
    EXIT_CODE=\$?
    
    if [ \$EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ Evaluation completed successfully!"
        echo "Results saved to: \$OUTPUT_DIR/"
    else
        echo ""
        echo "❌ Evaluation failed with exit code: \$EXIT_CODE"
    fi
else
    echo "❌ Error: Config file not found: \$CONFIG_FILE"
    exit 1
fi

echo "Job finished at \$(date)"
exit \$EXIT_CODE
EOF

    # Submit the job
    local job_id=$(sbatch "$temp_script" | awk '{print $4}')
    echo "  ✅ Submitted job $job_id: $job_name (Config #$config_num)"
    
    # Clean up temp script
    rm "$temp_script"
    
    return 0
}

# Submit jobs for all configs
SUBMITTED_COUNT=0

for config_file in "${CONFIG_FILES[@]}"; do
    config_num=$(get_config_num "$config_file")
    config_name=$(basename "$config_file" .yaml)
    job_name=$(get_job_name "$config_num")
    
    if submit_job "$config_file" "$config_num" "$job_name" "$DEVICE" "$INIT_METHODS" "$OUTPUT_DIR"; then
        ((SUBMITTED_COUNT++))
    else
        echo "  ❌ Failed to submit job for $config_name"
    fi
done

# Print summary
echo ""
echo "=========================================="
echo "Job Submission Summary"
echo "=========================================="
echo "Total configs: ${#CONFIG_FILES[@]}"
echo "Successfully submitted: $SUBMITTED_COUNT"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: logs/"
echo "Results will be saved to: $OUTPUT_DIR/"
echo ""

if [ $SUBMITTED_COUNT -eq ${#CONFIG_FILES[@]} ]; then
    echo "🎉 All jobs submitted successfully!"
    exit 0
else
    echo "⚠️  Some job submissions failed."
    exit 1
fi
