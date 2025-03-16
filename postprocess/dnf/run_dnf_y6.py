from __future__ import print_function
from glob import glob
import numpy as np
import sys
# import fnmatch
import y6dnf
import fitsio
import time
import gc

# parameters
nfilters = 4


def run_dnf_single_file(T, Terr, TZ, filename, outfile, cut=99):

    GALAXY = fitsio.read(filename, columns=['MAG_G', 'MAG_R', 'MAG_I', 'MAG_Z',
                                            'MAGERR_G', 'MAGERR_R', 'MAGERR_I', 'MAGERR_Z',
                                            'ID'])

    GALAXY = GALAXY[(GALAXY['MAG_G'] < cut)&(GALAXY['MAG_R'] < cut)&(GALAXY['MAG_I'] < cut)&(GALAXY['MAG_Z'] < cut)]

    Ngalaxies = len(GALAXY)
    print('Photo galaxies=', Ngalaxies)

    G = np.zeros((Ngalaxies, nfilters), dtype='double')
    Gerr = np.zeros((Ngalaxies, nfilters), dtype='double')

    G[:, 0] = GALAXY['MAG_G']
    G[:, 1] = GALAXY['MAG_R']
    G[:, 2] = GALAXY['MAG_I']
    G[:, 3] = GALAXY['MAG_Z']

    Gerr[:, 0] = GALAXY['MAGERR_G']
    Gerr[:, 1] = GALAXY['MAGERR_R']
    Gerr[:, 2] = GALAXY['MAGERR_I']
    Gerr[:, 3] = GALAXY['MAGERR_Z']

    # VALID=GALAXY
    V = G
    Verr = Gerr
    ####################################

    start = 0.0
    stop = 1.6
    step = 0.01

    zbins = np.arange(start, stop, step)

    # DNF'call
    chunksize = 200000
    nchunk = (Ngalaxies + chunksize - 1) // chunksize

    z_photo = np.zeros(Ngalaxies)
    z1 = np.zeros(Ngalaxies)
    zerr_e = np.zeros(Ngalaxies)

    for i in range(nchunk):
        print("working on chunk {0} out of {1}".format(i, nchunk), flush=True)
        ts = time.time()
        # z_photo,zerr_e,photozerr_param,photozerr_fit,Vpdf,z1,nneighbors,de1,d1,id1,C
        z_photo[i*chunksize:(i+1)*chunksize], zerr_e[i*chunksize:(i+1)*chunksize], _, _, Vpdf, z1[i*chunksize:(i+1)*chunksize], nneighbors, _,_,_,closestDistance = y6dnf.dnf(
            T, TZ, V[i*chunksize:(i+1)*chunksize], Verr[i*chunksize:(i+1)*chunksize], zbins, pdf=True, Nneighbors=80, bound=False, radius=2.0, magflux='mag', metric="DNF",coeff=True)
        print("Finished, using {0} s", time.time()-ts, flush=True)

    print("mean Nneighbors=", np.mean(nneighbors))

    out = np.zeros(Ngalaxies, dtype=np.dtype([('ID', np.int64), ('DNF_Z', float), ('DNF_ZN', float), ('DNF_ZSIGMA',float)]))
    out['ID'] = GALAXY['ID']
    out['DNF_Z'] = z_photo
    out['DNF_ZN'] = z1
    out['DNF_ZSIGMA'] = zerr_e

    fitsio.write(outfile, out, clobber=True)
    del out
    del z_photo
    del z1
    del zerr_e
    del G
    del Gerr
    del  GALAXY
    gc.collect()



if __name__ == '__main__':
    GALAXY = fitsio.read(sys.argv[1]) ##Training file
    fileglob = glob(sys.argv[2])
    if len(sys.argv) > 3:
        merge = True
    else:
        merge = False

    Ngalaxies = len(GALAXY)
    print('Train galaxies=', Ngalaxies)

    G = np.zeros((Ngalaxies, nfilters), dtype='double')
    Gerr = np.zeros((Ngalaxies, nfilters), dtype='double')

    G[:, 0] = GALAXY['mag_g']
    G[:, 1] = GALAXY['mag_r']
    G[:, 2] = GALAXY['mag_i']
    G[:, 3] = GALAXY['mag_z']

    Gerr[:, 0] = GALAXY['magerr_g']
    Gerr[:, 1] = GALAXY['magerr_r'] 
    Gerr[:, 2] = GALAXY['magerr_i']
    Gerr[:, 3] = GALAXY['magerr_z'] 

    Ntrain = Ngalaxies

    skip = 1
    T = G[::skip]
    Terr = Gerr[::skip]
    TZ = GALAXY['redshift'][::skip]

#    comm = MPI.COMM_WORLD
#    rank = comm.Get_rank()
#    size = comm.Get_size()

    if not merge:
        for filename in fileglob:
            outfile = filename.replace('fits', 'dnf.fits')
            run_dnf_single_file(T, Terr, TZ, filename, outfile)
    else:
        print(fileglob, flush=True)
        for i, filename in enumerate(fileglob):
            if i == 0:
                outfile = filename.split('.')
                outfile[-1] = 'combined.dnf'
                outfile = '.'.join(outfile)

            infile = filename.replace('fits', 'dnf.fits')
            dnf_data = fitsio.read(infile)

            f = fitsio.FITS(outfile, 'rw')
            if len(f) > 1:
                f[-1].append(dnf_data)
            else:
                f.write(dnf_data)

    sys.exit()
