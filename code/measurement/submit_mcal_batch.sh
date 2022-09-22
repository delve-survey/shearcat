#!/bin/bash

for ((i=0;i<10;i++))

do
echo $i

cd /scratch/midway2/chihway/delve_metacal
mkdir tile_${i}
cd tile_${i}
cp /project2/chihway/chihway/shearcat/code/measurement/download_tile.py ./.
cp /project2/chihway/chihway/shearcat/code/measurement/measure_mcal_batch.py ./.
cp /project2/chihway/chihway/shearcat/code/measurement/tile_DR3_1_1.csv ./.

python download_tile.py ${i}

echo "#!/bin/sh
#SBATCH -t 10:00:00
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --job-name=metacal_${i}
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30

python measure_mcal_batch.py ${i}

mv metacal_output_*fits /project2/chihway/data/decade/shearcat_v1/.
rm -rf /scratch/midway2/chihway/delve_metacal/tile_${i}/decade.ncsa.illinois.edu


">submit

sbatch submit

done

