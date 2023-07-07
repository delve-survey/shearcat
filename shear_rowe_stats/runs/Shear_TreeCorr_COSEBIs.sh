#!/bin/bash
#SBATCH --job-name TreeCorr_COSEBI_V2
#SBATCH --partition=broadwl
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --time=35:00:00
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

catalog=/project2/chihway/data/decade/metacal_test_20230427.hdf #/project2/chihway/data/decade/metacal_test_20230328.hdf

NOW=$( date '+%H:%M:%S' )
echo "Starting measure stage at $NOW"

python -u $ROWE_STATS_RUN_DIR/shear_treecorring.py --cat_path $catalog --Name ForCOSEBI_DESscales --min_angle 2.5 --max_angle 250 --nbins 1000 --bin_slop 0.1
