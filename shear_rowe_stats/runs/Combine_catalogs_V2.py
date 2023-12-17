
import numpy as np, healpy as hp
import fitsio
import pandas as pd
import os, shutil
from time import time
import glob
import h5py
from tqdm import tqdm


if __name__ == '__main__':
    
    
    #Get all files available
    filelist1 = glob.glob('/project/chihway/dhayaa/DECADE/rowe_stats/DR3_1_1/output*.fits')
    filelist2 = glob.glob('/project/chihway/dhayaa/DECADE/rowe_stats/DR3_1_2/output*.fits')
    filelist3 = glob.glob('/project/chihway/dhayaa/DECADE/rowe_stats/DR3_1_2_decn30/output*.fits')
    filelist4 = glob.glob('/project/chihway/dhayaa/DECADE/rowe_stats/DR3_1_2_Reproc/output*.fits')
     
    filelist = filelist1 + filelist2 + filelist3 + filelist4 #Concatenate the lists
    filelist = sorted(filelist); print("FILELIST SIZE:", len(filelist))
    filelist = np.unique(filelist); print("FILELIST SIZE:", len(filelist))
    
    print(filelist)
    #Open just one file to get column names
    columns = fitsio.FITS(filelist[0])[1].get_colnames()
    
    def get_column(filelist, column):
        
        return np.concatenate([fitsio.FITS(d)[1].read(columns=column) for d in tqdm(filelist, desc = column[5:])], axis = 0)
    
    
    def mask(filelist):
        
        M = []
#         EXCLUDE = fitsio.FITS('/home/dhayaa/Desktop/DECADE/shearcat/shear_rowe_stats/delve_exclude_20230725.fits')[1].read()
#         INCLUDE = fitsio.FITS('/project/chihway/dhayaa/DECADE/rowe_stats/FinalExpList_20230912.fits')[1].read()
        INCLUDE = fitsio.FITS('/project/chihway/dhayaa/DECADE/rowe_stats/FinalExpList_20231216.fits')[1].read()
        INCLUDE_UNIQUE_EXP = np.unique(INCLUDE['EXPNUM'])
        
        for f in tqdm(filelist, desc = "BUILD INCLUDE LIST  (20231216)"):
            
            EXP = fitsio.FITS(f)[1].read(columns='col0.EXPNUM')
            CCD = fitsio.FITS(f)[1].read(columns='col0.CCDNUM')
            
            #Find all images where EXPnum is in include list
            E = np.isin(EXP, INCLUDE_UNIQUE_EXP)
            
            #For all exposures that were selected above, find the corresponding CCDs to include and now
            #check if those CCD nums are in the catalog
            C = np.isin(CCD, INCLUDE['CCDNUM'][np.isin(INCLUDE['EXPNUM'], np.unique(EXP[E]))])

            #Final
            T = C & E
            
            #Only select objects that ARE in the above
            M.append(T)
#             M.append(np.invert(T))
            
        return np.concatenate(M) 
            
    
    with h5py.File('/scratch/midway3/dhayaa/star_psf_shapecat_20231216.hdf5', "w") as f:

        MASK = mask(filelist)
        
        #Create all columns you need
        for c in columns:
            if 'BAND' in c:
                f.create_dataset(c[5:], data = get_column(filelist, c).astype('S')[MASK], dtype = h5py.special_dtype(vlen=str))
            else:
                f.create_dataset(c[5:], data = get_column(filelist, c)[MASK])
                
        #Deredden quantities
        for name in ['SFD98', 'Planck13']:

            if name == 'SFD98':
                EXTINCTION = hp.read_map('/project/chihway/dhayaa/DECADE/Extinction_Maps/ebv_sfd98_nside_4096_ring_equatorial.fits')
                R_SFD98    = EXTINCTION[hp.ang2pix(4096, f['ra'][:], f['dec'][:], lonlat = True)]
                Ag, Ar, Ai, Az = R_SFD98*3.186, R_SFD98*2.140, R_SFD98*1.569, R_SFD98*1.196

            elif name == 'Planck13':
                EXTINCTION = hp.read_map('/project/chihway/dhayaa/DECADE/Extinction_Maps/ebv_planck13_nside_4096_ring_equatorial.fits')
                R_PLK13    = EXTINCTION[hp.ang2pix(4096, f['ra'][:], f['dec'][:], lonlat = True)]
                Ag, Ar, Ai, Az = R_PLK13*4.085, R_PLK13*2.744, R_PLK13*2.012, R_PLK13*1.533

            #Compute the extinction factors based on what band the measurement is
            A = np.zeros_like(f['ra']) + np.NaN
            b = f['BAND'][:].astype('U1')
            for band, ext in zip(['g', 'r', 'i', 'z'], [Ag, Ar, Ai, Az]):
                A = np.where(b == band, ext, A)
                
            assert np.isnan(A).sum() == 0, "Hmm. There are somehow %d NaNs in the extinction array" % np.isnan(A).sum()
                
                
            for c in ['FLUXERR_APER_8', 'FLUXERR_AUTO', 'FLUX_APER_8', 'FLUX_AUTO']:

                print(c + '_dered')
                arr = f[c][:] * 10**(A/2.5)
                f.create_dataset(c + '_DERED_' + name.upper(), data = arr)

            f.create_dataset('A_' + name.lower(), data = A)            

            
            
            
            