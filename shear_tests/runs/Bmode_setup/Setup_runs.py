

TEXT = """#!/bin/bash
#SBATCH --job-name DELVE_SimBModeRunner_%(INDEX)d
#SBATCH --partition=amd
#SBATCH --account=pi-chihway
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=36:00:00
#SBATCH --output=/home/dhayaa/DECADE/shearcat/shear_tests/runs/Bmode_setup/log_DELVE_SimBModeRunner_%(INDEX)d
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END

for i in {%(START)d..%(END)d}
do
    python -u /home/dhayaa/DECADE/shearcat/shear_tests/SimBModeRunner.py --DELVE --Npatch 150 \\
                                                                         --seed ${i} --Nind ${i} \\
                                                                         --Name "Cosmogrid_Test_20240918_N${i}"
done                               
"""


N_start = 1
N_end   = 150
N_per_batch = 5
N_batch = N_end // N_per_batch

for i in range(N_batch):
    
    args = {'INDEX' : i, 'START': i * N_per_batch + 1, 'END': (i+1) * N_per_batch + 1}
    
    with open(f'job_batch{i}.sh', 'w') as f:
        
        f.write(TEXT % args)