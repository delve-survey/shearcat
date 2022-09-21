
import sys
sys.path.append('/project2/chihway/chihway/measure_shear/')
sys.path.append('/project2/chihway/chihway/measure_shear/metacal')
sys.path.append('/project2/chihway/chihway/eastlake')

from _step import _run_metacal as run_metacal
import fitsio
import numpy as np
import yaml
import os

i = int(sys.argv[1])

metadata = np.genfromtxt('tile_DR3_1_1.csv', dtype='str', delimiter=",")[1:][i]

tile = metadata[0][2:-1]
path = metadata[1][2:-1]
p = path[-3:]
seed = 100

dir_meds = '/scratch/midway2/chihway/delve_metacal/tile_'+i+'/decade.ncsa.illinois.edu/deca_archive/'+path+'/meds/'

filename = [dir_meds+tile+'_r5918'+p+'_r_meds-delve.fits.fz',
            dir_meds+tile+'_r5918'+p+'_i_meds-delve.fits.fz',
            dir_meds+tile+'_r5918'+p+'_z_meds-delve.fits.fz']

with open('/project2/chihway/chihway/mcal_sim_test/runs/run_template/metacal.yaml', 'r') as fp:
     mcal_config = yaml.load(fp, Loader=yaml.Loader)

output = run_metacal(filename, seed, mcal_config) #seed can be an integer, for instance

fitsio.write('metacal_output_'+tile+'.fits', output, clobber=True)


