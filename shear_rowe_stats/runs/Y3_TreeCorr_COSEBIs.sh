#!/bin/bash
#SBATCH --job-name Y3tests_COSEBIs
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=40:00:00
#SBATCH --output=/home/dhayaa/Desktop/DECADE/shearcat/code/rowe_stats/runs/%x.log
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END


if [ "$USER" == "dhayaa" ]
then
    cd /home/dhayaa/Desktop/DECADE/shearcat/
    module load python
    conda activate shearDM
    source /home/dhayaa/Desktop/DECADE/bash_profile.sh
fi

for tag in None foreground badregions gold footprint mcalphotoz All
do
  python -u $ROWE_STATS_RUN_DIR/Y3tests_treecorring.py --cat_path /project2/chihway/dhayaa/DES_Catalogs/DESY3_MetacalCat_Mask_${tag}.npy \
                                                       --Name COSEBI_${tag} \
                                                       --min_angle 2.5 \
                                                       --max_angle 250 \
                                                       --nbins 1000 \
                                                       --bin_slop 1
done
