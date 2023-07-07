
'''
ratio of galaxy density to stellar density.
'''


if __name__ == '__main__':
    
    import numpy as np
    import h5py
    import healpy as hp
    import argparse

    my_parser = argparse.ArgumentParser()

    #Metaparams
    my_parser.add_argument('--NSIDE',     action='store', type = int, default = 1024)
    my_parser.add_argument('--Name',      action='store', type = str, required = True)
    my_parser.add_argument('--Output',    action='store', type = str, required = True)
    my_parser.add_argument('--StarCatalog',   action='store', type = str)
    my_parser.add_argument('--GalaxyCatalog', action='store', type = str)

    nside = 256

    star   = np.zeros(hp.nside2npix(nside))
    galaxy = np.zeros(hp.nside2npix(nside))

    with h5py.File(args['GalaxyCatalog'], 'r') as f:
        
        if 'ra' in f.keys():
            pix = hp.ang2pix(nside, f['ra'], f['dec'], lonlat = True)
        elif 'RA' in f.keys():
            pix = hp.ang2pix(nside, f['RA'], f['DEC'], lonlat = True)
    
    unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    galaxy[unique_pix] += np.bincount(idx_rep, weights = f['weight'])
    
    
    with h5py.File(args['StarCatalog'], 'r') as f:
        
        if 'ra' in f.keys():
            pix = hp.ang2pix(nside, f['ra'], f['dec'], lonlat = True)
        elif 'RA' in f.keys():
            pix = hp.ang2pix(nside, f['RA'], f['DEC'], lonlat = True)
            
    unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    star[unique_pix] += np.bincount(idx_rep)

    weight_map = np.zeros_like(star)
    weight_map[star != 0] = galaxy[star != 0]/star[star != 0]

    hp.write_map(args['Output'] + '/%s_stargalaxy_weightmap.fits' % args['Name'])
    
    