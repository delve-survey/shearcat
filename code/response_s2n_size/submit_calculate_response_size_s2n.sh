#!/bin/sh

for ((i=0;i<20;i++))
do
	
for ((j=0;j<20;j++))
do
	
echo "#!/bin/sh

#SBATCH -t 00:30:00
#SBATCH --partition=caslake
#SBATCH --account=pi-chihway
#SBATCH --job-name=response_${i}_${j}
#SBATCH --exclusive
#SBATCH --nodes=1

python calculate_response_size_s2n.py ${i} ${j}

">submit_${i}_${j}

chmod +x submit_${i}_${j}
sbatch submit_${i}_${j}

done
done
