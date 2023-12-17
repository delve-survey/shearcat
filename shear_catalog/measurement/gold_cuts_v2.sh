#! /bin/sh

meta=/project/chihway/chihway/shearcat/Tilelist/07112023/Tilelist_Reprocess_20231207.csv

cd /project/chihway/data/decade/coaddcat_v8
cp /project/chihway/chihway/shearcat/shear_catalog/measurement/download_cat_v2.py ./.
cp /project/chihway/chihway/shearcat/shear_catalog/measurement/gold_cuts_v2.py ./.
cp ${meta} ./.

for ((i=0; i<5881; i++))

do
echo $i

cd /project/chihway/data/decade/coaddcat_v8
python download_cat_v2.py ${i} ${meta}
python gold_cuts_v2.py ${i} ${meta}

done


