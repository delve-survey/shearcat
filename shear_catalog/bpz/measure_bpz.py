import numpy as np
import os
import subprocess as sp

bpz_run = '/project/chihway/chihway/DESC_BPZ/scripts/bpz.py'

metadata = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_1.txt', dtype='str', delimiter=",")[1:]

prior = 'sva1_weights'

for i in range(10): #len(metadata)): #change here to make short run if you wish  
    tile = metadata[i][0]
    print(i, tile)

    pars_file = '/project/chihway/data/decade/coaddcat_pz/DELVE_gold_'+str(tile)+'.pars'
    h5_file = '/project/chihway/data/decade/coaddcat_pz/gold_'+str(tile)+'.h5'
    
    if os.path.isfile(h5_file):            

        list_lines = ['COLUMNS    /project/chihway/chihway/DESC_BPZ/tests/CosmicShearPZ_BDF_Flux.columns\n'
                  , 'OUTPUT\t    /project/chihway/data/decade/coaddcat_pz/pz_'+str(prior)+'_DELVE_gold_'+str(tile)+'.h5\n'
                  , 'SPECTRA     CWWSB4.list\n'
                  , 'PRIOR\t    '+str(prior)+'\n'
                  , 'DZ          0.01\n'
                  , 'ZMIN        0.005\n'
                  , 'ZMAX        3.505\n'
                  , 'MAG         no\n'
                  , 'NEW_AB      no\n'
                  , 'MADAU\t    no #TURN OFF MADAU!!!!\n'
                  , 'EXCLUDE     none\n'
                  , 'CHECK       yes\n'
                  , '#ZC          1.0,2.0\n'
                  , '#FC          0.2,0.4\n'
                  , 'VERBOSE     no\n'
                  , '#INTERP      0\n'
                  , 'ODDS        0.68\n'
                  , 'PROBS      no\n'
                  , 'PROBS2     no\n'
                  , 'PROBS_LITE no\n'
                  , 'GET_Z       yes\n'
                  , 'INTERACTIVE yes\n'
                  , 'PLOTS       no\n'
                  , 'SAMPLING yes\n'
                  , 'NSAMPLES 1\n'
                  , 'SEED 42\n'
                  , '#ONLY_TYPE   yes\n']

        pars = open(pars_file, mode='w')
        pars.writelines(list_lines)
        pars.close()

        command = 'python -u '+str(bpz_run)+' '+str(h5_file)+'  -P '+str(pars_file)
        sp.run(command, shell = True)
        sp.run(f'rm '+h5_file, shell = True)
        sp.run(f'rm '+pars_file, shell = True)
    else: print(tile+' empty tile')


