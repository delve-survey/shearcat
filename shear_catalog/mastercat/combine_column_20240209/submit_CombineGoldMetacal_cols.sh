#!/bin/sh

for col in  badfrac id mcal_T_1m mcal_T_1p mcal_T_2m mcal_T_2p mcal_T_noshear mcal_T_ratio_1m mcal_T_ratio_1p mcal_T_ratio_2m mcal_T_ratio_2p mcal_T_ratio_noshear mcal_flags mcal_flux_1m mcal_flux_1p mcal_flux_2m mcal_flux_2p mcal_flux_err_1m mcal_flux_err_1p mcal_flux_err_2m mcal_flux_err_2p mcal_flux_err_noshear mcal_flux_noshear mcal_g_1m mcal_g_1p mcal_g_2m mcal_g_2p mcal_g_cov_1m mcal_g_cov_1p mcal_g_cov_2m mcal_g_cov_2p mcal_g_cov_noshear mcal_g_noshear mcal_psf_T_noshear mcal_psf_g_noshear mcal_s2n_1m mcal_s2n_1p mcal_s2n_2m mcal_s2n_2p mcal_s2n_noshear x y Ncutouts_raw

do

if [ ! -f "/project/chihway/data/decade/metacal_gold_columns_20240209/metacal_gold_combined_20240209_${col}.hdf" ] 
then

echo "#!/bin/sh
#SBATCH -t 10:00:00
#SBATCH --partition=amd
#SBATCH --account=pi-chihway
#SBATCH --job-name=combine_catalog_${col}
#SBATCH --exclusive
#SBATCH --nodes=1

python CombineGoldMetacal_cols.py 'mcal' ${col}

" > submit_${col}.sh

chmod +x submit_${col}.sh
sbatch submit_${col}.sh

fi
done


for col in COADD_OBJECT_ID RA DEC FLUX_AUTO_G FLUX_AUTO_R FLUX_AUTO_I FLUX_AUTO_Z FLUXERR_AUTO_G FLUXERR_AUTO_R FLUXERR_AUTO_I FLUXERR_AUTO_Z FLUX_RADIUS_G FLUX_RADIUS_R FLUX_RADIUS_I FLUX_RADIUS_Z BDF_FLUX_G BDF_FLUX_R BDF_FLUX_I BDF_FLUX_Z BDF_FLUX_ERR_G BDF_FLUX_ERR_R BDF_FLUX_ERR_I BDF_FLUX_ERR_Z BDF_T BDF_S2N
do

if [ ! -f "/project/chihway/data/decade/metacal_gold_columns_20240209/metacal_gold_combined_20240209_${col}.hdf" ]
then

echo "#!/bin/sh
#SBATCH -t 10:00:00
#SBATCH --partition=amd
#SBATCH --account=pi-chihway
#SBATCH --job-name=combine_catalog_${col}
#SBATCH --exclusive
#SBATCH --nodes=1

python CombineGoldMetacal_cols.py 'gold' ${col}

"> submit_${col}.sh

chmod +x submit_${col}.sh
sbatch submit_${col}.sh

fi
done




