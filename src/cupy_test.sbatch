#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=cupy
#SBATCH --output=cupy.out
#SBATCH --error=cupy.err
#SBATCH --time=0-00:15:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=28
#SBATCH --mem=242G

# Load Python modules
module load python/booth/3.8/3.8.5

srun python3 cupy_test.py  