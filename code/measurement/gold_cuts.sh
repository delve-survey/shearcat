#! /bin/sh

cd /project/chihway/data/decade/coaddcat_v1
cp /home/chihway/shearcat/code/measurement/download_cat.py ./.
cp /home/chihway/shearcat/code/measurement/gold_cuts.py ./.
cp /home/chihway/shearcat/Tilelist_DR3_1_1.csv ./.

for ((i=1000; i<2000; i++))

do
echo $i

cd /project/chihway/data/decade/coaddcat_v1
python download_cat.py ${i}
python gold_cuts.py ${i}

done


