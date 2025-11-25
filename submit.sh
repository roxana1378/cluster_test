#!/bin/bash
#
#SBATCH --mem-per-cpu 2000
#SBATCH -c 2
#SBATCH -J star_sim
#SBATCH -t 240:00
##SBATCH -o logs/%x_%j.out
##SBATCH -e logs/%x_%j.err
module load python/3.12.4
module load scipy-stack
source /home/roxana78/python_torch/bin/activate
srun python moons_nf.py
