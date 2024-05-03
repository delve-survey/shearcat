#!/bin/sh
#SBATCH -t 36:00:00
#SBATCH --partition=caslake
#SBATCH --account=pi-chihway
#SBATCH --job-name=combine_catalog
#SBATCH --exclusive
#SBATCH --nodes=1

python CombineGoldMetacal_mask_v2.py


