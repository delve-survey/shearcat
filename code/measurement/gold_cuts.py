
from astropy.io import fits
import numpy as np
import sys
import os

i = int(sys.argv[1])

metadata = np.genfromtxt('tile_DR3_1_1_v2.csv', dtype='str', delimiter=",")[1:][i]

tile = metadata[0][2:-1]
path = 'DEC_Taiga'+metadata[1][5:-6]
p = path[-4:-1]
print(tile, path, p)

#metadata = np.genfromtxt('tile_DR3_1_1.csv', dtype='str', delimiter=",")[1:][i]

#tile = metadata[0][2:-1]
#path = metadata[1][2:-1]
#p = path[-3:]

dir_cat = '/project2/chihway/data/decade/coaddcat_v1/decade.ncsa.illinois.edu/deca_archive/'+path+'/cat/'
dir_wavg = '/project2/chihway/data/decade/coaddcat_v1/decade.ncsa.illinois.edu/deca_archive/'+path+'/wavg/'

data_det = fits.open(dir_cat+tile+'_r5918'+p+'_det_cat.fits')[1].data
data_g = fits.open(dir_cat+tile+'_r5918'+p+'_g_cat.fits')[1].data
data_r = fits.open(dir_cat+tile+'_r5918'+p+'_r_cat.fits')[1].data
data_i = fits.open(dir_cat+tile+'_r5918'+p+'_i_cat.fits')[1].data
data_z = fits.open(dir_cat+tile+'_r5918'+p+'_z_cat.fits')[1].data

data_g_wavg = fits.open(dir_wavg+tile+'_r5918'+p+'_g_wavg.fits')[1].data
data_r_wavg = fits.open(dir_wavg+tile+'_r5918'+p+'_r_wavg.fits')[1].data
data_i_wavg = fits.open(dir_wavg+tile+'_r5918'+p+'_i_wavg.fits')[1].data
data_z_wavg = fits.open(dir_wavg+tile+'_r5918'+p+'_z_wavg.fits')[1].data


mask_SEflag = (data_g['FLAGS']<=3)*(data_r['FLAGS']<=3)*(data_i['FLAGS']<=3)*(data_z['FLAGS']<=3)
mask_IMAflag = (data_g['IMAFLAGS_ISO']==0)*(data_r['IMAFLAGS_ISO']==0)*(data_i['IMAFLAGS_ISO']==0)*(data_z['IMAFLAGS_ISO']==0)
extend_coadd = np.array(((data_i_wavg['WAVG_SPREAD_MODEL']+3*data_i_wavg['WAVG_SPREADERR_MODEL'])>0.005))*1 + np.array((data_i_wavg['WAVG_SPREAD_MODEL']+data_i_wavg['WAVG_SPREADERR_MODEL'])>0.003) *1 + np.array((data_i_wavg['WAVG_SPREAD_MODEL']-data_i_wavg['WAVG_SPREADERR_MODEL'])>0.002)*1

ra = data_det['ALPHAWIN_J2000']
dec = data_det['DELTAWIN_J2000']
flux_r = data_r['FLUX_AUTO']
flux_i = data_i['FLUX_AUTO']
flux_z = data_z['FLUX_AUTO']

np.savez('gold_mask_'+tile+'.npz', maskSE=np.array(mask_SEflag), maskIMA=np.array(mask_IMAflag), maskSG=np.array(extend_coadd), ra=ra, dec=dec, flux_r=flux_r, flux_i=flux_i, flux_z=flux_z)


os.system('rm -rf '+dir_cat)
os.system('rm -rf '+dir_wavg)

