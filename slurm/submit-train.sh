#!/bin/bash
#SBATCH --job-name=3DUnet
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH -o /home/users/train110/Nowcasting-with-PyEarthTools/slurm/output/%j.out
#SBATCH -e /home/users/train110/Nowcasting-with-PyEarthTools/slurm/error/%j.err

echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES}"

# Activate your PyEarthTools venv
source /home/users/train110/venv/pet_dev_nb_gpu_jasmin/bin/activate

# Make sure the pyearthtools_jasmin path is visible
export PYTHONPATH="/home/users/train110/pyearthtools_jasmin/src:$PYTHONPATH"

# Make math libraries single-threaded inside each process
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Help with HDF5 on network filesystems (for netCDF, xarray etc.)
export HDF5_USE_FILE_LOCKING=FALSE

# DDP / NCCL robustness: modern names
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Better crash traces
export PYTHONFAULTHANDLER=1

# Optional: put W&B logs somewhere explicit (you already log under wandb_logs in code)
export WANDB_DIR="/gws/ssde/j25a/mmh_storage/train110/wandb_logs"

# DDP setup for torchrun
export MASTER_ADDR="localhost"
export MASTER_PORT=$((12000 + RANDOM % 20000))

echo "Starting distributed training..."

torchrun --standalone --nproc_per_node=4 \
    /home/users/train110/Nowcasting-with-PyEarthTools/script/3D-Unet-DDP.py

echo "Training completed at $(date)"