#!/bin/bash

#meta=/project/chihway/chihway/shearcat/Tilelist/11072023/Tilelist_DR3_1_1_withASTROFIX.csv
meta=/project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_1.txt


for i in 550  #((i=0;i<6685;i++)) #6685

do
echo $i

j=`echo $i|awk '{print $1+2}'`
tile=`more $meta |head -$j|tail -1|sed s/','/' '/|awk '{print $1}'`

echo $tile

#if [ ! -f "/project/chihway/data/decade/shearcat_v2/metacal_output_${tile}.fits" ] || [ ! -f "/project/chihway/data/decade/shearcat_v2/ids_match_${tile}.npz" ]

#then

echo "file metacal_output_${tile}.fits does not exist!"

cd /scratch/midway3/chihway/delve_metacal
mkdir tile_${i}
cd tile_${i}
cp /project/chihway/chihway/shearcat/shear_catalog/measurement/download_tile.py ./.
cp /project/chihway/chihway/shearcat/shear_catalog/measurement/measure_mcal_batch.py ./.
cp ${meta} ./.

python download_tile.py ${i} ${meta}


echo "#!/bin/sh
#SBATCH -t 20:00:00
#SBATCH --partition=amd
##caslake
#SBATCH --account=pi-jfrieman
#SBATCH --job-name=metacal_${i}
#SBATCH --exclusive
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=28

python measure_mcal_batch.py ${i} ${meta}

cd /scratch/midway3/chihway/delve_metacal/tile_${i}

mv metacal_output_*fits /project/chihway/data/decade/shearcat_v2/.
mv *.npz /project/chihway/data/decade/shearcat_v2/.
rm -rf /scratch/midway3/chihway/delve_metacal/tile_${i}/decade.ncsa.illinois.edu
rm /scratch/midway3/chihway/delve_metacal/tile_${i}/*py
rm /scratch/midway3/chihway/delve_metacal/tile_${i}/*.csv

">submit

# comment to just check the file size 
sbatch submit

#else

echo "file metacal_output_${tile}.fits exist!"

#fi

done



