#!/bin/bash

#SBATCH --partition=milano --account lcls
#
#SBATCH --job-name=freqDirParallelTest
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.txt
#
#SBATCH --time=0-048:00:00
#
#SBATCH --mem=500G           # total memory per node
#
#SBATCH --exclusive

mpirun python /sdf/home/w/winnicki/sketchingParallelTests/parallelrun.py