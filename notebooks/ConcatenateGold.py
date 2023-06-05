import numpy as np
import astropy.io.fits as pf
import scipy
from scipy import interpolate
import h5py

def extProduction(BDF_T, BDF_S2N):
    
    x = [-3.       ,  0.79891862,  0.90845217,  0.98558583,  1.05791208,
         1.13603715,  1.22479487,  1.33572223,  1.48983602,  1.74124395,
         2.43187589,  6.        ] 
    y_1 = [0.028, 0.028, 0.008, 0.   , 0.004, 0.012, 0.012, 0.004, 0.012,
           0.024, 0.04 , 0.04 ]
    y_2 = [-0.028, -0.028, -0.04 , -0.032, -0.036, -0.032, -0.028, -0.016,
           -0.012,  0.008,  0.016,  0.016]
    y_3 = [-0.1  , -0.1  , -0.1  , -0.1  , -0.1  , -0.1  , -0.1  , -0.1  ,
           -0.012,  0.008,  0.016,  0.016]
    y_4 = [0.252, 0.252, 0.188, 0.14 , 0.096, 0.104, 0.052, 0.048, 0.04 ,
           0.052, 0.088, 0.088]

    f_array = [scipy.interpolate.interp1d(x, y_1, fill_value=-99, bounds_error=False),
               scipy.interpolate.interp1d(x, y_2, fill_value=-99, bounds_error=False),
               scipy.interpolate.interp1d(x, y_3, fill_value=-99, bounds_error=False),
               scipy.interpolate.interp1d(x, y_4, fill_value=-99, bounds_error=False)]

    x_data = np.log10(BDF_S2N)
    x_data = np.where(np.isfinite(x_data), x_data, x[0])
    y_data = BDF_T.copy()
    ext = np.tile(0, len(x_data))
    for f in f_array:
        selection = (y_data > f(x_data))
        ext += selection.astype(int)
    
    # Sentinel values
    selection = np.isclose(BDF_T, -9.999e+09) | np.isclose(BDF_S2N, -9.999e+09) | (BDF_S2N <= 0.)
    ext[selection] = -9

    return np.where(np.isfinite(ext), ext, -9)

## make gold mask

GOLD_MASK = []
SG = []

for i in range(81):
    print(i)
    if i<9:
        infile = pf.open('/scratch/midway2/chihway/gold_base_catalog_00000'+str(i+1)+'.fits')
    else: 
        infile = pf.open('/scratch/midway2/chihway/gold_base_catalog_0000'+str(i+1)+'.fits')
        
    bdf_t = infile[1].data['BDF_T']
    bdf_s2n = infile[1].data['BDF_S2N']   
    sg_bdf = extProduction(bdf_t, bdf_s2n)
    mask = (sg_bdf>=3) # check this
    GOLD_MASK.append(mask)
    SG.append(sg_bdf[mask])


def get_column(column):

    output = []
    for i in range(81):
        if i<9:
            infile = pf.open('/scratch/midway2/chihway/gold_base_catalog_00000'+str(i+1)+'.fits')[1].data
        else: 
            infile = pf.open('/scratch/midway2/chihway/gold_base_catalog_0000'+str(i+1)+'.fits')[1].data
            
        arr = infile[column][GOLD_MASK[i]]
            
        output.append(arr)

    return np.concatenate(output, axis = 0)


## store tile name in two pieces due to memory

Output1 = []
Output2 = []

for i in range(81):
    print(i)
    output1 = []
    output2 = []
    
    if i<9:
        infile = pf.open('/scratch/midway2/chihway/gold_base_catalog_00000'+str(i+1)+'.fits')[1].data
    else: 
        infile = pf.open('/scratch/midway2/chihway/gold_base_catalog_0000'+str(i+1)+'.fits')[1].data

    arr = infile['TILENAME'][GOLD_MASK[i]]
    for j in range(len(arr)):
        output1.append(int(arr[j][3:7]))
        output2.append(int(arr[j][-5:]))

    Output1.append(output1)
    Output2.append(output2)

Tilename1 = np.concatenate(Output1)
Tilename2 = np.concatenate(Output2)

## finally go through all the columns

path = '/project2/chihway/data/decade/gold_20230522_v2.hdf'

columns = ['COADD_OBJECT_ID', 'RA', 'DEC', 
           'MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z', 
           'MAGERR_AUTO_G', 'MAGERR_AUTO_R', 'MAGERR_AUTO_I', 'MAGERR_AUTO_Z', 
           'FLUX_RADIUS_G', 'FLUX_RADIUS_R', 'FLUX_RADIUS_I', 'FLUX_RADIUS_Z', 
           'BDF_FLUX_G', 'BDF_FLUX_R', 'BDF_FLUX_I', 'BDF_FLUX_Z', 'BDF_FLUX_Y', 
           'BDF_FLUX_ERR_G', 'BDF_FLUX_ERR_R', 'BDF_FLUX_ERR_I', 'BDF_FLUX_ERR_Z', 'BDF_FLUX_ERR_Y'] #'TILENAME', 

with h5py.File(path, "w") as f:

    for c in columns:
        print(c)
        f.create_dataset(c, data = get_column(c))

    # Now add ra_dec
    f.create_dataset('SG',  data = np.concatenate(SG))
    f.create_dataset('Tilename1',  data = Tilename1)
    f.create_dataset('Tilename2',  data = Tilename2)


