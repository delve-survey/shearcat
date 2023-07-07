#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --job-name=test_metacal
#SBATCH --exclusive

#conda activate shear_decade
python test_metacal.py 'DES1213-3457'
python test_metacal.py 'DES1224-3749'
python test_metacal.py 'DES1225-4206' 
#python test_metacal.py 'DES1213-3457'

python test_metacal.py 'DES1318-3623'
#python test_metacal.py 'DES1319-3457'



