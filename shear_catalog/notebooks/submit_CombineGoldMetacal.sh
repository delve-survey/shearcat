#!/bin/sh
#SBATCH -t 72:00:00
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --job-name=combine_catalog
#SBATCH --exclusive
#SBATCH --nodes=1

python CombineGoldMetacal_mask.py


