import numpy as np
import h5py

def define_cuts(h5py_file_path, version = 'noshear'):
    '''
    Takes in file path and returns indices of all metacal galaxies 
    that pass the cuts
    
    h5py_file_path:
                path to the h5py metacal catalog file
                
    version:
                the metacal version to place cuts on. 
                Can be 'noshear', '1p', '1m', '2p', '2m'.
    '''
    
    with h5py.File(h5py_file_path, 'r') as f:
    
        print("QUANTITIES: ", list(f.keys()))
        g1, g2  = np.array(f['mcal_g_%s' % version]).T

        mag_r   = 30 - 2.5*np.log10(np.array(f['mcal_flux_%s' % version])[:, 0])
        mag_i   = 30 - 2.5*np.log10(np.array(f['mcal_flux_%s' % version])[:, 1])
        mag_z   = 30 - 2.5*np.log10(np.array(f['mcal_flux_%s' % version])[:, 2])

        SNR     = np.array(f['mcal_s2n_%s' % version])
        Tratio  = np.array(f['mcal_T_ratio_%s' % version])
        T       = np.array(f['mcal_T_%s' % version])
        flags   = np.array(f['mcal_flags'])
        
        FLAGS_Foreground = np.array(f['FLAGS_FOREGROUND'])

        
        #Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)
        SNR_Mask   = (SNR > 10) & (SNR < 1000)
        Tratio_Mask= Tratio > 0.5
        T_Mask     = T < 10
        Flag_Mask  = flags == 0
        Other_Mask = np.invert((T > 2) & (SNR < 30)) & np.invert((np.log10(T) < (22.25 - mag_r)/3.5) & (g1**2 + g2**2 > 0.8**2))
        
        GOLD_Mask  = FLAGS_Foreground == 0 #From gold catalog
        SG_Mask    = np.array(f['sg_bdf']) >= 4 #Star-galaxy separator
        
        Color_Mask = ((18 < mag_i) & (mag_i < 23.5) & 
                      (15 < mag_r) & (mag_r < 26) & 
                      (15 < mag_z) & (mag_z < 26) & 
                      (-1.5 < mag_r - mag_i) & (mag_r - mag_i < 4) & 
                      (-1.5 < mag_i - mag_z) & (mag_i - mag_z < 4)
                     )
        
        del SNR, Tratio, T, flags, mag_i, mag_r, mag_z, g1, g2
        
        Final_Mask = SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Other_Mask & GOLD_Mask & SG_Mask & Color_Mask
        
        print("NAME, TOTAL, FRAC")
        
        Names = ['SNR', 'Tratio', 'T', 'Mcal_Flag', 'Others', 'Foreground', 'Star/Gal', 'Color', 'Final']
        for n, m in zip(Names,
                        [SNR_Mask, Tratio_Mask, T_Mask, Flag_Mask, Other_Mask, GOLD_Mask, SG_Mask, Color_Mask, Final_Mask]):
            
            print(n, np.sum(m), np.average(m))
        

    return np.where(Final_Mask)[0]


if __name__ == '__main__':
    
    #EXAMPLE RUN
    
    inds = define_cuts('/project/chihway/data/decade/metacal_gold_combined_20230613.hdf', version = 'noshear')
    print(inds)