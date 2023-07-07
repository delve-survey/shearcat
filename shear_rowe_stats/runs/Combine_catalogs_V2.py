#use this to measure the stars' and psf models' ellipticities

import numpy as np
import fitsio
import pandas as pd
import os, shutil
from time import time
import glob
import h5py
from tqdm import tqdm


if __name__ == '__main__':
    
    
    path = os.environ['ROWE_STATS_DIR']
    
    #Get all files available
    filelist = glob.glob(path + '/*.fits')
    filelist = sorted(filelist)
        
    #Open just one file to get column names
    columns = fitsio.FITS(filelist[0])[1].get_colnames()
    
#     datasets = [fitsio.FITS(f) for f in tqdm(filelist, desc = 'filelist')]
        
    def get_column(filelist, column):
        
        return np.concatenate([fitsio.FITS(d)[1].read(columns=column) for d in tqdm(filelist, desc = column[5:])], axis = 0)
    
    
    with h5py.File('/scratch/midway2/dhayaa/test.hdf5', "w") as f:
#     with h5py.File(os.path.join(path, 'star_psf_shapecat.hdf5'), "w") as f:

        #Create all columns you need
        for c in columns:
            if 'BAND' in c:
                f.create_dataset(c[5:], data = get_column(filelist, c).astype('S'), dtype = h5py.special_dtype(vlen=str))
            else:
                f.create_dataset(c[5:], data = get_column(filelist, c))
            