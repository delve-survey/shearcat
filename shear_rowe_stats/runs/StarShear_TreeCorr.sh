#!/bin/bash
#SBATCH --job-name StarShear
##SBATCH --partition=caslake
##SBATCH --account=pi-chihway
#SBATCH --partition=kicp
#SBATCH --account=kicp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=30:00:00
#SBATCH --output=/home/dhayaa/DECADE/shearcat/shear_rowe_stats/runs/%x.log
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END


if [ "$USER" == "dhayaa" ]
then
    cd /home/dhayaa/DECADE/shearcat/
    #module load python
    #conda activate /project/chihway/dhayaa/MyEnvs_Midway3/shear
    source /home/dhayaa/DECADE/mcal_sim_test/bash_profile.sh
fi

shearcatalog=/project/chihway/data/decade/metacal_gold_combined_20230613.hdf
psfcatalog=/project/chihway/dhayaa/DECADE/star_psf_shapecatalog_20230510.hdf5

export ROWE_STATS_RUN_DIR=/home/dhayaa/DECADE/shearcat/shear_rowe_stats/
export ROWE_STATS_DIR=/scratch/midway3/dhayaa/TEMP

NOW=$( date '+%H:%M:%S' )
echo "Starting measure stage at $NOW"

for SNR in 40;
do
    python -u $ROWE_STATS_RUN_DIR/star_shear_treecorring.py --gal_cat_path $shearcatalog \
                                                   --psf_cat_path $psfcatalog \
                                                   --Name Faint_d20230828_SNR${SNR} \
                                                   --min_angle 3 \
                                                   --max_angle 250 \
                                                   --nbins 25 \
                                                   --bin_slop 0.001 \
                                                   --m_min 16.5 \
                                                   --SNRCut ${SNR}


    python -u $ROWE_STATS_RUN_DIR/star_shear_treecorring.py --gal_cat_path $shearcatalog \
                                                   --psf_cat_path $psfcatalog \
                                                   --Name Bright_d20230828_SNR${SNR} \
                                                   --min_angle 3 \
                                                   --max_angle 250 \
                                                   --nbins 25 \
                                                   --bin_slop 0.001 \
                                                   --m_max 16.5 \
                                                   --SNRCut ${SNR}
done;
