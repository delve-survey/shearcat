#!/bin/bash
#SBATCH --job-name PSFColorDep
##SBATCH --partition=caslake
##SBATCH --account=pi-chihway
#SBATCH --partition=kicp
#SBATCH --account=kicp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=10:00:00
#SBATCH --output=/home/dhayaa/DECADE/PSFColorDep.log
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END

cd /home/dhayaa/DECADE/shearcat/
#module load python
#conda activate /project/chihway/dhayaa/MyEnvs_Midway3/shear
source /home/dhayaa/DECADE/mcal_sim_test/bash_profile.sh


shearcatalog=/project/chihway/data/decade/metacal_gold_combined_20230613.hdf
psfcatalog=/project/chihway/dhayaa/DECADE/star_psf_shapecatalog_20230510.hdf5
magcatalog=/project/chihway/dhayaa/DECADE/matched_star_psf_shapecatalog_20230630.npy


for SNR in 40 80 160;
do
	python -u utils/PSFColorDep.py --gal_cat_path $shearcatalog \
        	                       --psf_cat_path $psfcatalog \
                	               --psf_mag_path $magcatalog \
                        	       --Name d20230830_SNR${SNR} \
				       --SNRCut ${SNR}
done
