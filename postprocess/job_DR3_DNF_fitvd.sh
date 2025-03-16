#!/bin/bash
#SBATCH --job-name DNF
#SBATCH --output=/home/dhayaa/DECADE/shearcat/postprocess/log_%x
#SBATCH --partition=kicp
#SBATCH --account=kicp
#SBATCH --ntasks=96
#SBATCH --time=48:00:00
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END

source /home/dhayaa/setup_shear.sh
# module load openmpi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mpirun  --mca btl openib,self,vader \
        --mca btl_openib_allow_ib true \
        --mca btl_openib_warn_default_gid_prefix 0 \
        --mca btl_openib_warn_no_device_params_found 0 \
        --mca btl_tcp_port_min_v4 20000 --mca btl_tcp_port_range_v4 1000 \
        -np ${SLURM_NTASKS} python -m mpi4py /home/dhayaa/DECADE/shearcat/postprocess/DR3_DNF_fitvd.py --METACAL

