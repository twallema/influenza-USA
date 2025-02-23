#!/bin/bash
#SBATCH --job-name=incremental-calibration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00

# Submit as follows:
# sbatch --export=ALL,USE_ED_VISITS=False,INFORMED=True,SEASON="2014-2015" submit_job.sh

# Pin the number of cores for use in python calibration script
export NUM_CORES=$SLURM_CPUS_PER_TASK

# Load any necessary modules or dependencies
module load anaconda3

# Activate the virtual environment
conda activate INFLUENZA-USA

# Run your Python script
python calibrate_incremental-SIR_sequentialTwoStrain.py --use_ED_visits "$USE_ED_VISITS" --informed "$INFORMED" --season "${SEASON}"

# Deactivate the virtual environment
conda deactivate