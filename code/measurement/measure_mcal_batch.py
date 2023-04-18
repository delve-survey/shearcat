
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

metadata = np.genfromtxt('/home/chihway/shearcat/Tilelist_DR3_1_1.csv', dtype='str', delimiter=",")[1:][i]

tile = metadata[0]
path = 'DEC_Taiga'+metadata[1][3:-5]

p = path[-4:-1]
seed = 100

print(tile, path, p)

dir_meds = '/scratch/midway3/chihway/delve_metacal/tile_'+str(i)+'/decade.ncsa.illinois.edu/deca_archive/'+path+'/meds/'

filename = [dir_meds+tile+'_r5918'+p+'_r_meds-delve.fits.fz',
            dir_meds+tile+'_r5918'+p+'_i_meds-delve.fits.fz',
            dir_meds+tile+'_r5918'+p+'_z_meds-delve.fits.fz']

with open('/home/chihway/mcal_sim_test/runs/run_template/metacal.yaml', 'r') as fp:
     mcal_config = yaml.load(fp, Loader=yaml.Loader)

output = run_metacal(filename, seed, mcal_config) #seed can be an integer, for instance

fitsio.write('metacal_output_'+tile+'.fits', output, clobber=True)


X = meds.MEDS(filename[0])
np.savez('ids_match_'+tile+'.npz', ids=X['id'])

