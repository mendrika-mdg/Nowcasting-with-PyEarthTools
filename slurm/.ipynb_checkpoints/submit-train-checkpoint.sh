#!/bin/bash
#SBATCH --job-name=3DUnet                     # Job name
#SBATCH --partition=orchid                    # Partition to run on
#SBATCH --account=orchid                      # Account
#SBATCH --qos=orchid                          # QoS level
#SBATCH --nodes=1                             # Single node
#SBATCH --ntasks-per-node=4                   # torchrun spawns 4 processes (1 per GPU)
#SBATCH --gres=gpu:4                          # 4 GPUs per node
#SBATCH --cpus-per-task=4                     # 4 CPU threads per process
#SBATCH --mem=256G                            # Total memory allocation
#SBATCH --time=48:00:00                       # Wall time
#SBATCH --exclude=gpuhost006,gpuhost015       # Exclude bad GPU nodes
#SBATCH -o /home/users/train110/Nowcasting-with-PyEarthTools/slurm/output/%j.out
#SBATCH -e /home/users/train110/Nowcasting-with-PyEarthTools/slurm/error/%j.err

echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES}"

# Activate virtual environment
source /home/users/train110/venv/pet_dev_nb_gpu_jasmin/bin/activate

# ---------------------------------------------------------------------
# Torch DDP configuration
# ---------------------------------------------------------------------
export MASTER_ADDR="localhost"                  # Local master for torchrun
export MASTER_PORT=$((12000 + RANDOM % 20000))  # Random port to avoid conflicts
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK     # Set OpenMP threads per task


# ---------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------
echo "Starting distributed training..."

torchrun --standalone --nproc_per_node=4 /home/users/train110/Nowcasting-with-PyEarthTools/script/3D-Unet-DDP.py

echo "Training completed at $(date)"