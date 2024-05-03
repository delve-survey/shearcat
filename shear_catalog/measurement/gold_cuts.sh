#! /bin/sh

meta=/project/chihway/chihway/shearcat/Tilelist/11072023/Tilelist_DR3_1_2_withASTROFIX.csv

cd /project/chihway/data/decade/coaddcat_v4
cp /project/chihway/chihway/shearcat/shear_catalog/measurement/download_cat.py ./.
cp /project/chihway/chihway/shearcat/shear_catalog/measurement/gold_cuts.py ./.
cp ${meta} ./.

for ((i=100; i<13000; i++))

do
echo $i

cd /project/chihway/data/decade/coaddcat_v4
python download_cat.py ${i} ${meta}
python gold_cuts.py ${i} ${meta}

done


