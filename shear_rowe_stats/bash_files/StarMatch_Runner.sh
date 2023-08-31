#!/bin/bash
#SBATCH --job-name Starmatch_Runner
#SBATCH --partition=caslake
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=10:00:00
#SBATCH --output=/home/dhayaa/DECADE/StarMatcher.log
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END

cd /home/dhayaa/DECADE/shearcat/
#module load python
#conda activate /project/chihway/dhayaa/MyEnvs_Midway3/shear
source /home/dhayaa/DECADE/mcal_sim_test/bash_profile.sh

python -u utils/StarMatcher.py
