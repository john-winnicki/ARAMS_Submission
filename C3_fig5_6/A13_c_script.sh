#!/bin/bash

#SBATCH --partition=milano --account lcls
#
#SBATCH --job-name=freqDirTest
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.txt
#
#SBATCH --time=0-024:00:00
#
#SBATCH --ntasks=1
#
# SBATCH --exclusive

## SBATCH --mem-per-cpu=20G

mpirun -np 1 python /sdf/home/w/winnicki/sketchingTrueRun/run.py
