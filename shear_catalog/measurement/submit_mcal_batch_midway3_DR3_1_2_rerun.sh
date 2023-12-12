#!/bin/bash

#meta=/project/chihway/chihway/shearcat/Tilelist/11072023/Tilelist_DR3_1_2_withASTROFIX.csv
#/project/chihway/dhayaa/DECADE/ReprocessTilelist_20231106.csv
meta=/project/chihway/chihway/shearcat/Tilelist/07112023/Tilelist_Reprocess_20231207.csv


for i in [ 241 875 886 2227 2345 2381 3992 4280 ]  #((i=0;i<4000;i++)) 

do
echo $i

j=`echo $i|awk '{print $1+2}'`
tile=`more $meta |head -$j|tail -1|sed s/','/' '/|awk '{print $1}'`

echo $tile

if [ ! -f "/project/chihway/data/decade/shearcat_v5/metacal_output_${tile}.fits" ] || [ ! -f "/project/chihway/data/decade/shearcat_v5/ids_match_${tile}.npz" ]

then

echo "file metacal_output_${tile}.fits does not exist!"

cd /scratch/midway3/chihway/delve_metacal
mkdir tile_${i}
cd tile_${i}
cp /project/chihway/chihway/shearcat/shear_catalog/measurement/download_tile.py ./.
cp /project/chihway/chihway/shearcat/shear_catalog/measurement/measure_mcal_batch.py ./.
cp $meta ./.

python download_tile.py ${i} ${meta}


echo "#!/bin/sh
#SBATCH -t 20:00:00
#SBATCH --partition=caslake
##SBATCH --partition=broadwl
#SBATCH --account=pi-chihway
#SBATCH --job-name=metacal_${i}
##SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
##SBATCH --ntasks-per-node=28

python measure_mcal_batch.py ${i} ${meta}

cd /scratch/midway3/chihway/delve_metacal/tile_${i}

mv metacal_output_*fits /project/chihway/data/decade/shearcat_v5/.
mv *.npz /project/chihway/data/decade/shearcat_v5/.
rm -rf /scratch/midway3/chihway/delve_metacal/tile_${i}/decade.ncsa.illinois.edu
rm /scratch/midway3/chihway/delve_metacal/tile_${i}/*py
rm /scratch/midway3/chihway/delve_metacal/tile_${i}/*.csv

">submit

# check file size
sbatch submit
#ls -l decade.ncsa.illinois.edu/deca_archive/DEC_Taiga/multiepoch/delve/r*/DES*/*/meds/*fz >size
#rm decade.ncsa.illinois.edu/deca_archive/DEC_Taiga/multiepoch/delve/r*/DES*/*/meds/*fz

else

echo "file metacal_output_${tile}.fits exist!"

fi

done



