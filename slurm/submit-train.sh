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

# Quick environment sanity check
which python
python -c "import site_archive_jasmin as saj; print('ROOT_DIRECTORIES:', saj.ROOT_DIRECTORIES)"
python -c "import torch; print('Torch:', torch.__version__)"

# DDP setup
export MASTER_ADDR="localhost"
export MASTER_PORT=$((12000 + RANDOM % 20000))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting distributed training..."

torchrun --standalone --nproc_per_node=4 \
    /home/users/train110/Nowcasting-with-PyEarthTools/script/3D-Unet-DDP.py

echo "Training completed at $(date)"
