#!/bin/bash
#SBATCH --job-name MakeCat
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
    module load python -u
    conda activate shearDM
    source /home/dhayaa/Desktop/DECADE/bash_profile.sh
fi

catalog=$ROWE_STATS_DIR/star_psf_shapecat.hdf5

NOW=$( date '+%H:%M:%S' )
echo "Starting measure stage at $NOW"

python -u /home/dhayaa/Desktop/DECADE/shearcat/code/rowe_stats/runs/Make_Master_Catalog.py

#python -u /home/dhayaa/Desktop/DECADE/shearcat/code/rowe_stats/runs/Combine_catalogs_V2.py

#python -u /home/dhayaa/Desktop/DECADE/shearcat/code/rowe_stats/runs/Combine_catalogs_Legacy.py
