
import numpy as np
import os
import sys

meta = 'DR3_1_2_rerun_check.txt'

for i in range(10):
    metadata = np.genfromtxt(meta, dtype='str', delimiter=",")[1:][i]

    tile = metadata[1]
    path = metadata[2][:-6]

    print(tile, path)

    #command = 'wget --user=decade --password=decaFil3s  --recursive --no-parent --auth-no-challenge  https://decade.ncsa.illinois.edu/deca_archive/'+path+'/meds/'

    #os.system(command)

    command = 'wget --user=decade --password=decaFil3s  --recursive --no-parent --auth-no-challenge  https://decade.ncsa.illinois.edu/deca_archive/'+path+'/cat/'

    os.system(command)


