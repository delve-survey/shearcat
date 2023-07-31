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


    def match(self, nside = 8192*32):  

        hpix = hp.ang2pix(nside, self.ra, self.dec, lonlat = True)

        unique_hpix, inds, unique_counts = np.unique(hpix, return_inverse = True, return_counts = True)

        print("TOTAL:", len(hpix))
        print("UNIQUE MATCHED:", len(unique_hpix))
        print("MAX REPEATS:", np.max(unique_counts))
        print("MIN REPEATS:", np.min(unique_counts))

        return unique_hpix, inds, unique_counts
    
    
    def weighted_average(self, A, weights = None):

        wgt = np.bincount(self.inds, weights = weights)
        sm  = np.bincount(self.inds, weights = A*weights)        
        
        avg = sm/wgt

        return avg


if __name__ == '__main__':

    with h5py.File('/project/chihway/dhayaa/DECADE/star_psf_shapecatalog_20230510.hdf5', 'r') as f:

        ra   = np.array(f['ra']).astype(np.float32)
        dec  = np.array(f['dec']).astype(np.float32)
        
        hpix = hp.ang2pix(8192 * 32, ra, dec, lonlat = True)
        unique_hpix, inds, unique_counts = np.unique(hpix, return_inverse = True, return_counts = True)
        
        out  = np.zeros([3, hpix.size], dtype = np.float32) + np.NaN
        
        del hpix, unique_counts
        
        #Get selections for individual bands first
        BAND  = np.array(f['BAND']).astype('U1')
        Masks = [np.where(BAND == 'r')[0], np.where(BAND == 'i')[0], np.where(BAND == 'z')[0]]
        
        flux = np.array(f['FLUX_AUTO'])
        SNR  = flux/np.array(f['FLUXERR_AUTO'])
        ZP   = np.array(f['MAGZP'])
        mag  = ZP - 2.5 * np.log10(flux)
        
        del flux, ZP, BAND
        
        #This helps avoid nan errors later down the line
        #We set the mag to 0 but we don't really use it 
        #since the SNR will also be zero
        SNR  = np.where(SNR < 0, 0, SNR)
        mag  = np.where(np.isfinite(mag), mag, 0)
        
        print("LOADED RA, DEC, SNR")
        
        for i in tqdm(range(len(Masks)), desc = 'Matching bands'):
            
            m = Masks[i]

            Band_Matcher = StarMatch(ra = ra[m],  dec = dec[m])
            weighted_mag = Band_Matcher.weighted_average(mag[m], weights = SNR[m]) #Average all quantities together
                
            Masks[i] = None #This just "deletes" the mask in here.
            
            weighted_mag_map    = np.zeros(unique_hpix.size) + np.NaN
            tmp, inds_a, inds_b = np.intersect1d(unique_hpix, Band_Matcher.UniquePix, return_indices = True)
            
            print("Matched index size:", len(tmp))
            
            weighted_mag_map[inds_a] = weighted_mag[inds_b]; del weighted_mag
           
            out[i, :] = weighted_mag_map[inds]; del weighted_mag_map
            
            

    del Band_Matcher, ra, dec, SNR, Masks

    np.save('/project/chihway/dhayaa/DECADE/matched_star_psf_shapecatalog_20230630.npy', out)