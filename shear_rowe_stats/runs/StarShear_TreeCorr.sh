#!/bin/bash
#SBATCH --job-name StarShear
#SBATCH --partition=bigmem2
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --time=30:00:00
#SBATCH --output=/home/dhayaa/Desktop/DECADE/shearcat/code/rowe_stats/runs/%x.log
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END


if [ "$USER" == "dhayaa" ]
then
    cd /home/dhayaa/Desktop/DECADE/shearcat/
    module load python
    conda activate shearDM
    source /home/dhayaa/Desktop/DECADE/mcal_sim_test/bash_profile.sh
fi

shearcatalog=/project/chihway/data/decade/metacal_gold_combined_20230613.hdf
psfcatalog=/project/chihway/dhayaa/DECADE/star_psf_shapecatalog_20230510.hdf5

NOW=$( date '+%H:%M:%S' )
echo "Starting measure stage at $NOW"

python -u $ROWE_STATS_RUN_DIR/star_shear_treecorring.py --gal_cat_path $shearcatalog \
                                                   --psf_cat_path $psfcatalog \
                                                   --Name Faint_d03072023 \
                                                   --min_angle 3 \
                                                   --max_angle 250 \
                                                   --nbins 25 \
                                                   --bin_slop 0.001 \
                                                   --m_min 16.5


python -u $ROWE_STATS_RUN_DIR/star_shear_treecorring.py --gal_cat_path $shearcatalog \
                                                   --psf_cat_path $psfcatalog \
                                                   --Name Bright_d03072023 \
                                                   --min_angle 3 \
                                                   --max_angle 250 \
                                                   --nbins 25 \
                                                   --bin_slop 0.001 \
                                                   --m_max 16.5
