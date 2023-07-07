#!/bin/sh
#SBATCH -t 72:00:00
#SBATCH --partition=kicpaa
#SBATCH --account=kicpaa
#SBATCH --job-name=concatenate_gold
#SBATCH --exclusive
#SBATCH --nodes=1

python ConcatenateGold.py


