#!/bin/sh

# DR3_1_1 6684
# DR3_1_2 10170

for ((i=0;i<=65;i++))
do

echo "#!/bin/sh

#SBATCH -t 10:00:00
#SBATCH --partition=caslake
#SBATCH --account=pi-chihway
#SBATCH --job-name=bpz_${i}
#SBATCH --exclusive
#SBATCH --nodes=1

python measure_bpz.py ${i} /project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_1.txt 0

">submit_${i}

chmod +x submit_${i}
sbatch submit_${i}

done



echo "#!/bin/sh

#SBATCH -t 10:00:00
#SBATCH --partition=caslake
#SBATCH --account=pi-chihway
#SBATCH --job-name=bpz_last
#SBATCH --exclusive
#SBATCH --nodes=1

python measure_bpz.py 66 /project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_1.txt 6684

">submit_last

chmod +x submit_last
sbatch submit_last





