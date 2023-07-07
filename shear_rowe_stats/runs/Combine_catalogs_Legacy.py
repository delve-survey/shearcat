#use this to measure the stars' and psf models' ellipticities

import numpy as np
from astropy.io import fits
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
    columns = fits.open(filelist[0])[1].data.dtype.names
    
    with h5py.File('/scratch/midway2/dhayaa/test2.hdf5', "w") as f:

        #Create all columns you need
        for c in columns:
            if 'BAND' not in c:
                f.create_dataset(c[5:], data = [], chunks=(10**5,), maxshape = (None,))
            else:
                f.create_dataset(c[5:], data = [], chunks=(10**5,), maxshape = (None,), dtype = h5py.special_dtype(vlen=str))

        #Helper function. Just improves readability
        #Appends new_data array into existing dataset
        def add_data(dataset, new_data):

            dataset.resize(dataset.shape[0] + len(new_data), axis=0)
            dataset[-len(new_data):] = new_data

        #Open mcal files iteratively
        for i in tqdm(range(len(filelist))):

            #Get dataset
            d = fits.open(filelist[i])[1].data
            
            #Append to columns
            for c in columns:
                if 'BAND' in c:
                    add_data(f[c[5:]], d[c].astype('S'))
                else:
                    add_data(f[c[5:]], d[c])
                    
                    
            #print(f['EXPNUM'], d['col0.EXPNUM'])
