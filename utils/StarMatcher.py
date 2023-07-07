import numpy as np, pandas as pd
import h5py
from tqdm import tqdm
import healpy as hp
import os


import argparse


class StarMatch(object):

    def __init__(self, ra, dec):

        self.ra  = ra
        self.dec = dec

        output    = self.match()
        self.inds = output[1]
        self.UniquePix = output[0]


    def match(self, nside = 8096*32):  

        hpix = hp.ang2pix(nside, self.ra, self.dec, lonlat = True)

        unique_hpix, inds, unique_counts = np.unique(hpix, return_inverse = True, return_counts = True)

        print("TOTAL:", len(hpix))
        print("UNIQUE MATCHED:", len(hpix))
        print("MAX REPEATS:", np.max(unique_counts))
        print("MIN REPEATS:", np.min(unique_counts))

        return unique_hpix, inds, unique_counts
    
    
    def weighted_average(self, A, weights = None):

        wgt = np.bincount(self.inds, weights = weights)
        sm  = np.bincount(self.inds, weights = A*weights)
        
        
        
        avg = wgt/sm
#         avg = avg[wgt != 0]

        return avg


if __name__ == '__main__':

    with h5py.File('/project/chihway/dhayaa/DECADE/star_psf_shapecatalog_20230510.hdf5', 'r') as f:

        keys = ['ra', 'dec', 'FLUX_AUTO',
                'g1_star_hsm', 'g2_star_hsm', 'T_star_hsm', 
                'g1_model_hsm','g2_model_hsm','T_model_hsm', 
                'IMAFLAGS_ISO']

        N = 1_000_000_000_000
        
        #Get selections for individual bands first
        BAND = np.array(f['BAND'][:N]).astype('U1')

        Masks    = [np.where(BAND == 'r')[0], np.where(BAND == 'i')[0], np.where(BAND == 'z')[0]]
        Catalogs = []

        ra  = np.array(f['ra'][:N])
        dec = np.array(f['dec'][:N])

        SNR = np.array(f['FLUX_AUTO'][:N])/np.array(f['FLUXERR_AUTO'][:N])
        
        print("LOADED RA, DEC, SNR")
        
        for m in Masks:

            Matcher = StarMatch(ra = ra[m],  dec = dec[m])

            out = pd.DataFrame()

            out['inds'] = Matcher.UniquePix #The healpix pixel this star belongs to

            for k in keys:

                out[k] = Matcher.weighted_average(f[k][:N][m], weights = SNR[m]) #Average all quantities together
                
            out['FLUXERR_AUTO'] = np.sqrt(Matcher.weighted_average(f['FLUXERR_AUTO'][:N][m]**2, weights = SNR[m]))

            Catalogs.append(out)
        
    del Matcher, out, ra, dec, SNR, Masks, BAND

    #Combine all bands
    Catalog = Catalogs[0]
    Catalog = Catalog.join(Catalogs[1], on = 'inds', how = 'inner', lsuffix = 'r', rsuffix = 'i')
    Catalog = Catalog.join(Catalogs[2], on = 'inds', how = 'inner', lsuffix = '',  rsuffix = 'z')

    
    with h5py.File('/project/chihway/dhayaa/DECADE/matched_star_psf_shapecatalog_20230630.hdf', 'w') as f:
        
        for k in keys + ['inds', 'FLUXERR_AUTO']:
            f.create_dataset(k, data = Catalog[k])
    