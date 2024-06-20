#!/bin/bash
#SBATCH --partition=brook
#SBATCH --job-name=read_gpu_stats
#SBATCH --output=rolling_output_nojobnumber.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

# Check if Nvidia SMI is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: Nvidia SMI is not installed on this node."
    exit 1
fi

# Function to read GPU model
read_gpu_model() {
    gpu_model=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    echo "GPU Model: $gpu_model"
}

# Function to log GPU usage for each GPU on the node
log_gpu_usage() {
  local gpu_ids=(${CUDA_VISIBLE_DEVICES//,/ })
  for gpu_id in "${gpu_ids[@]}"; do
    nvidia-smi -i ${gpu_id} -lms=1 --query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits >> logs/gpu_usage_node${SLURM_NODEID}_gpu${gpu_id}.log &
  done
}

# Main script
read_gpu_model
log_gpu_usage &  # Run the logging function in the background

# Run the benchmark in the background and detach it from the SLURM job control
srun python3 resnet_multi.py >> logs/training_output_${SLURM_JOB_ID}.log

# Disown the background process to prevent SLURM from waiting for it
disown -h %+

# Now, exit the SLURM job immediately after finishing the benchmark
exit 0
