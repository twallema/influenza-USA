#!/bin/bash
#SBATCH --job-name=hierarchical_training
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --partition=defq

# Load any necessary modules or dependencies
module load anaconda3

# Activate the virtual environment
conda activate INFLUENZA-USA

# Run your Python script
python hierarchical_training-SIR_oneStrain.py

# Deactivate the virtual environment
conda deactivate