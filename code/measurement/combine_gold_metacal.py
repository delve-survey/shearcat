
import numpy as np
import sys
import os
sys.path.append('.conda/envs/shear/lib/python3.9/site-packages/')
import astropy.io.fits as pf

metadata = np.genfromtxt('tile_DR3_1_1_v2.csv', dtype='str', delimiter=",")[1:]

from astropy.table import Table, vstack
CAT = []

shear_dir = '/project2/chihway/data/decade/shearcat_v1/'
coadd_dir = '/project2/chihway/data/decade/coaddcat_v1/'

for i in range(2000):
    tile = metadata[i][0][2:-1]
#     print(tile)
    
    if os.path.exists(shear_dir+'metacal_output_'+tile+'.fits') and os.path.exists(coadd_dir+'gold_mask_'+tile+'.npz'):
        shear_cat = pf.open(shear_dir+'metacal_output_'+tile+'.fits')[1].data
        shear_id = np.load(shear_dir+'ids_match_'+tile+'.npz',allow_pickle=True)
        gold_mask = np.load(coadd_dir+'gold_mask_'+tile+'.npz',allow_pickle=True)
        gold_mask_full = gold_mask['maskSE']*gold_mask['maskIMA']*(gold_mask['maskSG']>=2) 
        # still using rough star/galaxy separation
        shear_id_masked_gold = shear_id['ids'][gold_mask_full]
        mask_joint = np.in1d(shear_cat['id'], shear_id_masked_gold)
        mask_joint_invert = np.in1d(shear_id_masked_gold, shear_cat['id'][mask_joint])
        ra = gold_mask['ra'][gold_mask_full][mask_joint_invert]
        dec = gold_mask['dec'][gold_mask_full][mask_joint_invert]
        flux_i = gold_mask['flux_i'][gold_mask_full][mask_joint_invert]
        flux_r = gold_mask['flux_r'][gold_mask_full][mask_joint_invert]
        flux_z = gold_mask['flux_z'][gold_mask_full][mask_joint_invert]

        tb = Table.read(shear_dir+'metacal_output_'+tile+'.fits', format='fits')[mask_joint]

        tb.add_columns([ra, dec, flux_i, flux_r, flux_z], names=['ra', 'dec', 'flux_i', 'flux_r', 'flux_z'])
        print(i, len(tb))
        CAT.append(tb)
    else:
        print('tile missing '+tile)


new = vstack(CAT)
print(len(new))
print(len(new)*1.0/(2000*0.5*60*60))
print(new)
new.write('/scratch/midway3/chihway/metacal_test_2000tiles_20230106.fits', overwrite=True)



