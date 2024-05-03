
from astropy.io import fits
import numpy as np
import sys
import os
import h5py

i = int(sys.argv[1])
meta = sys.argv[2]

metadata = np.genfromtxt(meta, dtype='str', delimiter=",")[1:][i]

tile = metadata[0]
path = 'DEC'+metadata[1][3:-5]
p = path[-4:-1]
proc_id = path[-22:-18]
#proc_id2 = '6015'
proc_id2 = '6128'
print(tile, path, p, proc_id, "======== enter gold_cut.py =========")

output_path='/project/chihway/data/decade/coaddcat_v4/'

dir_cat = output_path+'decade.ncsa.illinois.edu/deca_archive/'+path+'/cat/'
#dir_wavg = '/project/chihway/data/decade/coaddcat_v3/decade.ncsa.illinois.edu/deca_archive/'+path+'/wavg/'
dir_sof = output_path+'decade.ncsa.illinois.edu/deca_archive/DEC_Taiga/multiepoch/shear_SOF_DR3/r'+proc_id2+'/'+tile+'/p01/sof/'
#dir_sof = '/project/chihway/data/decade/coaddcat_v3/decade.ncsa.illinois.edu/deca_archive/DEC_Taiga/multiepoch/shear_SOF_DR3/r6128/'+tile+'/p01/sof/'


data_det = fits.open(dir_cat+tile+'_r'+proc_id+p+'_det_cat.fits')[1].data
data_g = fits.open(dir_cat+tile+'_r'+proc_id+p+'_g_cat.fits')[1].data
data_r = fits.open(dir_cat+tile+'_r'+proc_id+p+'_r_cat.fits')[1].data
data_i = fits.open(dir_cat+tile+'_r'+proc_id+p+'_i_cat.fits')[1].data
data_z = fits.open(dir_cat+tile+'_r'+proc_id+p+'_z_cat.fits')[1].data

data_sof = fits.open(dir_sof+tile+'_r'+proc_id2+'p01_sof.fits')[1].data

#data_g_wavg = fits.open(dir_wavg+tile+'_r5918'+p+'_g_wavg.fits')[1].data
#data_r_wavg = fits.open(dir_wavg+tile+'_r5918'+p+'_r_wavg.fits')[1].data
#data_i_wavg = fits.open(dir_wavg+tile+'_r5918'+p+'_i_wavg.fits')[1].data
#data_z_wavg = fits.open(dir_wavg+tile+'_r5918'+p+'_z_wavg.fits')[1].data


mask_SEflag = (data_g['FLAGS']<=3)*(data_r['FLAGS']<=3)*(data_i['FLAGS']<=3)*(data_z['FLAGS']<=3)
mask_IMAflag = (data_g['IMAFLAGS_ISO']==0)*(data_r['IMAFLAGS_ISO']==0)*(data_i['IMAFLAGS_ISO']==0)*(data_z['IMAFLAGS_ISO']==0)
mask_NITER = (data_g['NITER_MODEL']>0)*(data_r['NITER_MODEL']>0)*(data_i['NITER_MODEL']>0)*(data_z['NITER_MODEL']>0)
mask_GOLD = mask_SEflag*mask_IMAflag*mask_NITER

#extend_coadd = np.array(((data_i_wavg['WAVG_SPREAD_MODEL']+3*data_i_wavg['WAVG_SPREADERR_MODEL'])>0.005))*1 + np.array((data_i_wavg['WAVG_SPREAD_MODEL']+data_i_wavg['WAVG_SPREADERR_MODEL'])>0.003) *1 + np.array((data_i_wavg['WAVG_SPREAD_MODEL']-data_i_wavg['WAVG_SPREADERR_MODEL'])>0.002)*1

ra = data_det['ALPHAWIN_J2000']
dec = data_det['DELTAWIN_J2000']

flux_g = data_g['FLUX_AUTO']
flux_r = data_r['FLUX_AUTO']
flux_i = data_i['FLUX_AUTO']
flux_z = data_z['FLUX_AUTO']
fluxerr_g = data_g['FLUXERR_AUTO']
fluxerr_r = data_r['FLUXERR_AUTO']
fluxerr_i = data_i['FLUXERR_AUTO']
fluxerr_z = data_z['FLUXERR_AUTO']

flux_radius_g = data_g['FLUX_RADIUS']
flux_radius_r = data_r['FLUX_RADIUS']
flux_radius_i = data_i['FLUX_RADIUS']
flux_radius_z = data_z['FLUX_RADIUS']

flux_bdf = data_sof['bdf_flux']
fluxerr_bdf = data_sof['bdf_flux_err']
bdf_t = data_sof['bdf_T']
bdf_s2n = data_sof['bdf_s2n']
sof_id = data_sof['id']

# save h5 files
with h5py.File(output_path+'gold_'+tile+'.hdf5', "w") as f:
    f.create_dataset('RA', data = ra)
    f.create_dataset('DEC', data = dec)
    f.create_dataset('GOLD_MASK', data = mask_GOLD)
    f.create_dataset('FLUX_AUTO_G', data = flux_g)
    f.create_dataset('FLUX_AUTO_R', data = flux_r)
    f.create_dataset('FLUX_AUTO_I', data = flux_i)
    f.create_dataset('FLUX_AUTO_Z', data = flux_z)
    f.create_dataset('FLUXERR_AUTO_G', data = fluxerr_g)
    f.create_dataset('FLUXERR_AUTO_R', data = fluxerr_r)
    f.create_dataset('FLUXERR_AUTO_I', data = fluxerr_i)
    f.create_dataset('FLUXERR_AUTO_Z', data = fluxerr_z)
    f.create_dataset('FLUX_RADIUS_G', data = flux_radius_g)
    f.create_dataset('FLUX_RADIUS_R', data = flux_radius_r)
    f.create_dataset('FLUX_RADIUS_I', data = flux_radius_i)
    f.create_dataset('FLUX_RADIUS_Z', data = flux_radius_z)
    f.create_dataset('FLUX_BDF_G', data = flux_bdf[:,0][::-1])
    f.create_dataset('FLUX_BDF_R', data = flux_bdf[:,1][::-1])
    f.create_dataset('FLUX_BDF_I', data = flux_bdf[:,2][::-1])
    f.create_dataset('FLUX_BDF_Z', data = flux_bdf[:,3][::-1])
    f.create_dataset('FLUXERR_BDF_G', data = fluxerr_bdf[:,0][::-1])
    f.create_dataset('FLUXERR_BDF_R', data = fluxerr_bdf[:,1][::-1])
    f.create_dataset('FLUXERR_BDF_I', data = fluxerr_bdf[:,2][::-1])
    f.create_dataset('FLUXERR_BDF_Z', data = fluxerr_bdf[:,3][::-1])
    f.create_dataset('BDF_T', data = bdf_t[::-1])
    f.create_dataset('BDF_S2N', data = bdf_s2n[::-1])
    f.create_dataset('SOF_ID', data = sof_id[::-1])


#np.savez('gold_mask_'+tile+'.npz', maskSE=np.array(mask_SEflag), maskIMA=np.array(mask_IMAflag), maskSG=np.array(extend_coadd), ra=ra, dec=dec, flux_r=flux_r, flux_i=flux_i, flux_z=flux_z)


os.system('rm -rf '+dir_cat)
os.system('rm -rf '+dir_sof)

