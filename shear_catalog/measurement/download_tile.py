
import numpy as np
import os
import sys

i = int(sys.argv[1])
meta = sys.argv[2]

metadata = np.genfromtxt(meta, dtype='str', delimiter=",")[1:][i]

tile = metadata[0]
path = 'DEC_'+metadata[1][4:-5]

print(tile, path)



command = 'wget --user=decade --password=decaFil3s  --recursive --no-parent --auth-no-challenge  https://decade.ncsa.illinois.edu/deca_archive/'+path+'/meds/'

os.system(command)

