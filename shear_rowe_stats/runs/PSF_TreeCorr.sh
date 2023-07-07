#!/bin/bash
#SBATCH --job-name TreeCorr
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

catalog=$ROWE_STATS_DIR/star_psf_shapecat.hdf5

NOW=$( date '+%H:%M:%S' )
echo "Starting measure stage at $NOW"

for i in r i z;
do
    python -u $ROWE_STATS_RUN_DIR/psf_treecorring.py --cat_path $catalog --band ${i} --Name ${i}_band --SNRCut 40
done

#Have to do g separately
python -u -u $ROWE_STATS_RUN_DIR/psf_treecorring.py --cat_path $catalog --band 'g'   --Keepgband --Name 'g_band'   --SNRCut 40

python -u $ROWE_STATS_RUN_DIR/psf_treecorring.py --cat_path $catalog --band 'ALL' --Keepgband --Name 'All_band' --SNRCut 40

python -u $ROWE_STATS_RUN_DIR/psf_treecorring.py --cat_path $catalog --band 'ALL' --Name 'riz_band' --SNRCut 40

python -u $ROWE_STATS_RUN_DIR/psf_treecorring.py --cat_path $catalog --band 'ALL' --Name 'riz_band_SNR20' --SNRCut 20


#SNR cut on just r-band
python -u $ROWE_STATS_RUN_DIR/psf_treecorring.py --cat_path $catalog --band 'r' --Name 'r_band_SNR50' --SNRCut 50
python -u $ROWE_STATS_RUN_DIR/psf_treecorring.py --cat_path $catalog --band 'r' --Name 'r_band_SNR30' --SNRCut 30
python -u $ROWE_STATS_RUN_DIR/psf_treecorring.py --cat_path $catalog --band 'r' --Name 'r_band_SNR20' --SNRCut 20
