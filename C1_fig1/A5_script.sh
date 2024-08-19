#!/bin/bash

#SBATCH --partition=milano --account lcls
#
#SBATCH --job-name=freqDirTest
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.txt
#
#SBATCH --time=0-20:00:00
#
# SBATCH --exclusive
#
## SBATCH --ntasks=1
## SBATCH --mem-per-cpu=32G

# python /sdf/home/w/winnicki/papertests_20240717/testme.py top true 0.2
# python /sdf/home/w/winnicki/papertests_20240717/testme.py top false 0.2
# python /sdf/home/w/winnicki/papertests_20240717/testme.py top true 1.0
# python /sdf/home/w/winnicki/papertests_20240717/testme.py top false 1.0

# python /sdf/home/w/winnicki/papertests_20240717/testme.py mid true 0.2
# python /sdf/home/w/winnicki/papertests_20240717/testme.py mid false 0.2
# python /sdf/home/w/winnicki/papertests_20240717/testme.py mid true 1.0
# python /sdf/home/w/winnicki/papertests_20240717/testme.py mid false 1.0

# python /sdf/home/w/winnicki/papertests_20240717/testme.py bot true 0.2
# python /sdf/home/w/winnicki/papertests_20240717/testme.py bot false 0.2
# python /sdf/home/w/winnicki/papertests_20240717/testme.py bot true 1.0
# python /sdf/home/w/winnicki/papertests_20240717/testme.py bot false 1.0