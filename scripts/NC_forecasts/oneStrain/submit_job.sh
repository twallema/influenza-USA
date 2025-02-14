#!/bin/bash
#SBATCH --job-name=hierarchical_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=72:00:00

# Pin the number of cores for use in python calibration script
export NUM_CORES=$SLURM_CPUS_PER_TASK

# Load any necessary modules or dependencies
module load anaconda3

# Activate the virtual environment
conda activate INFLUENZA-USA

# Run your Python script
python hierarchical_training-SIR_oneStrain.py

# Deactivate the virtual environment
conda deactivate