#!/bin/bash
#SBATCH --job-name ShearTests
#SBATCH --partition=kicp
#SBATCH --account=kicp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=30:00:00
#SBATCH --output=/home/dhayaa/DECADE/shearcat/shear_tests/runs/%x.log
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END


if [ "$USER" == "dhayaa" ]
then
    source /home/dhayaa/DECADE/mcal_sim_test/bash_profile.sh
    conda activate /project/chihway/dhayaa/MyEnvs_Midway3/shear
fi

shearcatalog=/project/chihway/data/decade/metacal_gold_combined_20230613.hdf
psfcatalog=/project/chihway/dhayaa/DECADE/star_psf_shapecatalog_20230510.hdf5

RUN_DIR=/home/dhayaa/DECADE/shearcat/shear_tests

NOW=$( date '+%H:%M:%S' )
echo "Starting measure stage at $NOW"

python -u $RUN_DIR/ShearTestRunner.py --psf_cat "/project/chihway/dhayaa/DECADE/star_psf_shapecatalog_20230510.hdf5" \
                                      --galaxy_cat "/project/chihway/data/decade/metacal_gold_combined_20231002.hdf" \
                                      --psf_cat_inds "/scratch/midway3/dhayaa/SHEARTEST/psf_inds.npy" \
                                      --galaxy_cat_inds "/scratch/midway3/dhayaa/SHEARTEST/shear_inds.npy" \
                                      --output_path "/scratch/midway3/dhayaa/SHEARTEST/" \
                                      --sim_Cls_path "/scratch/midway3/dhayaa/SHEARTEST/Cls.npy"