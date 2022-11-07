#!/bin/bash

for ((i=0;i<1000;i++))

do
echo $i

j=`echo $i|awk '{print $1+2}'`
tile=`more tile_DR3_1_1_v2.csv |head -$j|tail -1|sed s/,/' '/|sed s/"b'"/' '/|sed s/"'"/' '/|awk '{print $1}'`
echo $tile

if [ ! -f "/project/chihway/data/decade/shearcat_v1/metacal_output_${tile}.fits" ]

then

echo "file metacal_output_${tile}.fits does not exist!"

cd /scratch/midway3/chihway/delve_metacal
mkdir tile_${i}
cd tile_${i}
cp /home/chihway/shearcat/code/measurement/download_tile.py ./.
cp /home/chihway/shearcat/code/measurement/measure_mcal_batch.py ./.
cp /home/chihway/shearcat/code/measurement/tile_DR3_1_1_v2.csv ./.

python download_tile.py ${i}


echo "#!/bin/sh
#SBATCH -t 10:00:00
#SBATCH --partition=caslake
#SBATCH --account=pi-kadrlica
#SBATCH --job-name=metacal_${i}
#SBATCH --exclusive
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=28

python measure_mcal_batch.py ${i}

cd /scratch/midway3/chihway/delve_metacal/tile_${i}

mv metacal_output_*fits /project/chihway/data/decade/shearcat_v1/.
mv *.npz /project/chihway/data/decade/shearcat_v1/.
rm -rf /scratch/midway3/chihway/delve_metacal/tile_${i}/decade.ncsa.illinois.edu
rm /scratch/midway3/chihway/delve_metacal/tile_${i}/*py
rm /scratch/midway3/chihway/delve_metacal/tile_${i}/tile_DR3_1_1_v2.csv

">submit

sbatch submit

else

echo "file metacal_output_${tile}.fits exist!"

fi

done



