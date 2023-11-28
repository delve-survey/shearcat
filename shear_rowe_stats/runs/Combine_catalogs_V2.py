
import numpy as np
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
    filelist = sorted(filelist)
    
    #Open just one file to get column names
    columns = fitsio.FITS(filelist[0])[1].get_colnames()
    
    def get_column(filelist, column):
        
        return np.concatenate([fitsio.FITS(d)[1].read(columns=column) for d in tqdm(filelist, desc = column[5:])], axis = 0)
    
    
    def mask(filelist):
        
        M = []
#         EXCLUDE = fitsio.FITS('/home/dhayaa/Desktop/DECADE/shearcat/shear_rowe_stats/delve_exclude_20230725.fits')[1].read()
        INCLUDE = fitsio.FITS('/home/dhayaa/Desktop/DECADE/FinalExpList.fits')[1].read()
        INCLUDE_UNIQUE_EXP = np.unique(INCLUDE['EXPNUM'])
        for f in tqdm(filelist, desc = "BUILD INCLUDE LIST  (20230910)"):
            
            CCD = fitsio.FITS(f)[1].read(columns='col0.EXPNUM')
            EXP = fitsio.FITS(f)[1].read(columns='col0.CCDNUM')
            
            #Find all images where EXPnum is in include list
            E = np.in1d(EXP, INCLUDE_UNIQUE_EXP)
            
            #For all exposures that were selected above, find the corresponding CCDs to include and now
            #check if those CCD nums are in the catalog
            C = np.in1d(CCD, INCLUDE['CCDNUM'][np.in1d(INCLUDE['EXPNUM'], np.unique(EXP[E]))])

            #Final
            T = C & E
            
            #Only select objects that ARE in the above
            M.append(T)
#             M.append(np.invert(T))
            
        return np.concatenate(M) 
            
    
    with h5py.File('/scratch/midway3/dhayaa/star_psf_shapecat_20231121.hdf5', "w") as f:

        MASK = mask(filelist)
        
        #Create all columns you need
        for c in columns:
            if 'BAND' in c:
                f.create_dataset(c[5:], data = get_column(filelist, c).astype('S')[MASK], dtype = h5py.special_dtype(vlen=str))
            else:
                f.create_dataset(c[5:], data = get_column(filelist, c)[MASK])
            
