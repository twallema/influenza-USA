#!/bin/bash
#SBATCH --job-name=hierarchical_training
#SBATCH --output=hierarchical_training.out
#SBATCH --error=hierarchical_training.err
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=defq

# Load any necessary modules or dependencies
module load anaconda3

# Activate the virtual environment
conda activate INFLUENZA-USA

# Run your Python script
python myscript.py

# Deactivate the virtual environment
conda deactivate