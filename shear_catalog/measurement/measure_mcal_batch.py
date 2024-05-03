
import sys
sys.path.append('/home/chihway/measure_shear/')
sys.path.append('/home/chihway/measure_shear/metacal')
sys.path.append('/home/chihway/eastlake')

from _step import _run_metacal as run_metacal
import fitsio
import numpy as np
import yaml
import os
import meds

i = int(sys.argv[1])
meta = sys.argv[2]

#metadata = np.genfromtxt('/home/chihway/shearcat/Tilelist_DR3_1_1.csv', dtype='str', delimiter=",")[1:][i]
metadata = np.genfromtxt(meta, dtype='str', delimiter=",")[1:][i]


tile = metadata[0]
#path = 'DEC_Taiga'+metadata[1][3:-5]
path = 'DEC'+metadata[1][3:-5]

p = path[-4:-1]
seed = 100

print(tile, path, p)

dir_meds = '/scratch/midway3/chihway/delve_metacal/tile_'+str(i)+'/decade.ncsa.illinois.edu/deca_archive/'+path+'/meds/'

#DR3_1_1
if os.path.isfile(dir_meds+tile+'_r5918'+p+'_r_meds-delve.fits.fz'):
    filename = [dir_meds+tile+'_r5918'+p+'_r_meds-delve.fits.fz',
            dir_meds+tile+'_r5918'+p+'_i_meds-delve.fits.fz',
            dir_meds+tile+'_r5918'+p+'_z_meds-delve.fits.fz']

if os.path.isfile(dir_meds+tile+'_r6101'+p+'_r_meds-delve.fits.fz'):
    filename = [dir_meds+tile+'_r6101'+p+'_r_meds-delve.fits.fz',
            dir_meds+tile+'_r6101'+p+'_i_meds-delve.fits.fz',
            dir_meds+tile+'_r6101'+p+'_z_meds-delve.fits.fz']

#DR3_1_1/2 new
if os.path.isfile(dir_meds+tile+'_r6235'+p+'_r_meds-delve.fits.fz'):
    filename = [dir_meds+tile+'_r6235'+p+'_r_meds-delve.fits.fz',
            dir_meds+tile+'_r6235'+p+'_i_meds-delve.fits.fz',
            dir_meds+tile+'_r6235'+p+'_z_meds-delve.fits.fz']

#DR3_1_2
if os.path.isfile(dir_meds+tile+'_r6050'+p+'_r_meds-delve.fits.fz'):
    filename = [dir_meds+tile+'_r6050'+p+'_r_meds-delve.fits.fz',
                dir_meds+tile+'_r6050'+p+'_i_meds-delve.fits.fz',
                dir_meds+tile+'_r6050'+p+'_z_meds-delve.fits.fz']

#DR3_1_2 astrofix
if os.path.isfile(dir_meds+tile+'_r6117'+p+'_r_meds-delve.fits.fz'):
    filename = [dir_meds+tile+'_r6117'+p+'_r_meds-delve.fits.fz',
                dir_meds+tile+'_r6117'+p+'_i_meds-delve.fits.fz',
                dir_meds+tile+'_r6117'+p+'_z_meds-delve.fits.fz']

#DR3_1_2 reprocess 
if os.path.isfile(dir_meds+tile+'_r6321'+p+'_r_meds-delve.fits.fz'):
    filename = [dir_meds+tile+'_r6321'+p+'_r_meds-delve.fits.fz',
                dir_meds+tile+'_r6321'+p+'_i_meds-delve.fits.fz',
                dir_meds+tile+'_r6321'+p+'_z_meds-delve.fits.fz']


#DR3_1 reprocess 
if os.path.isfile(dir_meds+tile+'_r6352'+p+'_r_meds-delve.fits.fz'):
    filename = [dir_meds+tile+'_r6352'+p+'_r_meds-delve.fits.fz',
                dir_meds+tile+'_r6352'+p+'_i_meds-delve.fits.fz',
                dir_meds+tile+'_r6352'+p+'_z_meds-delve.fits.fz']


#DR3_1 reprocess 
if os.path.isfile(dir_meds+tile+'_r6353'+p+'_r_meds-delve.fits.fz'):
    filename = [dir_meds+tile+'_r6353'+p+'_r_meds-delve.fits.fz',
                dir_meds+tile+'_r6353'+p+'_i_meds-delve.fits.fz',
                dir_meds+tile+'_r6353'+p+'_z_meds-delve.fits.fz']

with open('/home/chihway/mcal_sim_test/runs/run_template/metacal_production.yaml', 'r') as fp:
     mcal_config = yaml.load(fp, Loader=yaml.Loader)

output = run_metacal(filename, seed, mcal_config) #seed can be an integer, for instance

fitsio.write('metacal_output_'+tile+'.fits', output, clobber=True)


X = meds.MEDS(filename[0])
np.savez('ids_match_'+tile+'.npz', ids=X['id'])

