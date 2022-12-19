#!/bin/sh

#984948

#for exp in 1085416 1002805 512337 165848 166445 195525 185489 512334 

for exp in 998588

do

FILES="/project2/kicp/secco/delve/rowe_stats_files/exp${exp}/r/*.fits.fz"
for f in $FILES
do

name=`echo ${f}|sed s/"_immasked.fits.fz"/""/` 
name2=${name: -24}

echo ${name} ${name2}

echo "#!/bin/sh

#SBATCH -t 00:20:00
#SBATCH --partition=broadwl
#SBATCH --account=pi-chihway
#SBATCH --job-name=psfex_${name2}
#SBATCH --exclusive
#SBATCH --nodes=1

./rerun_psfex_cc.csh ${name} ${name2} 80

"> submit_${name2}

sbatch submit_${name2}

done

done
