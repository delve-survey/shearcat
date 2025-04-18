#!/bin/bash

for ((i=0;i<10;i++))

do
echo $i


cd /scratch/midway2/chihway/delve_metacal
mkdir tile_${i}
cd tile_${i}
cp /project2/chihway/chihway/shearcat/code/measurement/download_tile.py ./.
cp /project2/chihway/chihway/shearcat/code/measurement/measure_mcal_batch.py ./.
cp /project2/chihway/chihway/shearcat/code/measurement/tile_DR3_1_1_v2.csv ./.

python download_tile.py ${i}


echo "#!/bin/sh
#SBATCH -t 10:00:00
#SBATCH --partition=broadwl
#SBATCH --account=pi-chihway
#SBATCH --job-name=metacal_${i}
#SBATCH --exclusive
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=28

python measure_mcal_batch.py ${i}

mv metacal_output_*fits /project2/chihway/data/decade/shearcat_v1/.
mv *.npz /project2/chihway/data/decade/shearcat_v1/.
rm -rf /scratch/midway2/chihway/delve_metacal/tile_${i}/decade.ncsa.illinois.edu
rm /scratch/midway2/chihway/delve_metacal/tile_${i}/*py
rm /scratch/midway2/chihway/delve_metacal/tile_${i}/tile_DR3_1_1_v2.csv

">submit

sbatch submit

done

