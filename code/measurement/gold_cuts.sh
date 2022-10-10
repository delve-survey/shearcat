#! /bin/sh

cd /project2/chihway/data/decade/coaddcat_v1
cp /project2/chihway/chihway/shearcat/code/measurement/download_cat.py ./.
cp /project2/chihway/chihway/shearcat/code/measurement/gold_cuts.py ./.
cp /project2/chihway/chihway/shearcat/code/measurement/tile_DR3_1_1.csv ./.

for ((i=0;i<100;i++))

do
echo $i

cd /project2/chihway/data/decade/coaddcat_v1
python download_cat.py ${i}
python gold_cuts.py ${i}

done


