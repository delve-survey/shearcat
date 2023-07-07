#!/bin/bash
#SBATCH --job-name Starmatch_Runner
#SBATCH --partition=kicp
#SBATCH --account=kicp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --time=10:00:00
#SBATCH --output=/home/dhayaa/Desktop/DECADE/StarMatcher.log
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END

cd /home/dhayaa/Desktop/DECADE/shearcat/
module load python
conda activate shearDM
source /home/dhayaa/Desktop/DECADE/mcal_sim_test/bash_profile.sh

python -u utils/StarMatcher.py
