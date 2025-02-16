#!/bin/bash
#SBATCH --job-name=hierarchical_training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00

# Pin the number of cores for use in python calibration script
export NUM_CORES=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# Load any necessary modules or dependencies
module load anaconda3

# Activate the virtual environment
conda activate INFLUENZA-USA

# Run the Python script using MPI
mpirun -np $NUM_CORES python hierarchical_training-SIR_sequentialTwoStrain.py

# Deactivate the virtual environment
conda deactivate