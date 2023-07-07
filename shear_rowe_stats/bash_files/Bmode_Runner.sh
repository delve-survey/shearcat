#!/bin/bash
#SBATCH --job-name Bmode_Runner
##SBATCH --partition=broadwl
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
##SBATCH --ntasks-per-node=40
#SBATCH --time=30:00:00
#SBATCH --output=/home/dhayaa/Desktop/DECADE/BmodeRunner.log
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END

cd /home/dhayaa/Desktop/DECADE/shearcat/
module load python
conda activate shearDM
source /home/dhayaa/Desktop/DECADE/mcal_sim_test/bash_profile.sh

python -u utils/Bmode.py --Name DECADE_Aggressive --Output /scratch/midway2/dhayaa/TEMP/ --njobs 4 --Nrands 1000 --DECADE