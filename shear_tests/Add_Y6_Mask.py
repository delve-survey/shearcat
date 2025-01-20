import healpy as hp, healsparse as hsp
import numpy as np


MASK = hp.read_map('/project/chihway/dhayaa/DECADE/Foreground_Masks/GOLD_Ext0.2_Star5_MCs2.fits', dtype = int)

#Found in /global/cfs/projectdirs/des/monroy/Y6A2/maglim/joint_mask/maglim_joint_lss-shear_mask_nside16384_NEST_v4.hsp.gz
Y6   = hsp.HealSparseMap.read('/scratch/midway3/dhayaa/maglim_joint_lss-shear_mask_nside16384_NEST_v4.hsp.gz')
Y6   = Y6.generate_healpix_map(nside = 4096, nest = False) > 0 #pixels that have any DES Y6 info in them

MASK = MASK | np.where(Y6, 4096, 0)


hp.write_map('/project/chihway/dhayaa/DECADE/Foreground_Masks/GOLD_Ext0.2_Star5_MCs2_DESY6.fits', MASK, overwrite = True, dtype = np.int16)
