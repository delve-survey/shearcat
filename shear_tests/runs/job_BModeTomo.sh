#!/bin/bash
#SBATCH --job-name BmodeTomo
#SBATCH --partition=kicp
#SBATCH --account=kicp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=36:00:00
#SBATCH --output=/home/dhayaa/DECADE/shearcat/shear_tests/runs/BmodeTomo.log
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END

if [ "$USER" == "dhayaa" ]
then
    conda activate /project/chihway/dhayaa/MyEnvs_Midway3/shear
fi

RUN_DIR=/home/dhayaa/DECADE/shearcat/shear_tests

python -u $RUN_DIR/BmodeTomoRunner.py --galaxy_cat "/project/chihway/data/decade/metacal_gold_combined_20240209.hdf" \
                                      --output_path "/scratch/midway3/dhayaa/SHEARTESTS_20240901/BmodeTomo" \
                                      --sim_Cls_path "/project/chihway/dhayaa/DECADE/cosmosis/Lucas_files/Kappa_Cls.txt" \
                                      --Ncov 500
