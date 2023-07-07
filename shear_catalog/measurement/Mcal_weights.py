import numpy as np
import h5py
from scipy import interpolate

def compute_weights(SN, ToverTPSF, e1, e2, mcal_response):
    SN, ToverTPSF = None, None

    e1, e2 = None, None

    SN_2sigma_max        = np.percentile(SN, 97.5)
    ToverTPSF_2sigma_max = np.percentile(ToverTPSF, 97.5)

    bins_SN         = np.geomspace(10,  SN_2sigma_max,        20 + 1)
    bins_ToverTPSF  = np.geomspace(0.5, ToverTPSF_2sigma_max, 20 + 1)

    bins_SN        = bins_SN.append(np.max(SN))
    bins_ToverTPSF = bins_ToverTPSF.append(np.max(ToverTPSF))

    weight = np.zeros([bin_SN.size - 1, bin_ToverTPSF.size - 1])

    weight_per_galaxy = np.zeros_like(e1)

    for i in range(bins_SN.size - 1):

        SN_mask = (SN > bins_SN[i]) & (SN < bins_SN[i])

        for j in range(bins_ToverTPSF.size - 1):

            ToverTPSF_mask = (bins_ToverTPSF > bins_ToverTPSF[i]) & (bins_ToverTPSF < bins_ToverTPSF[i])

            total_mask  = SN_mask & ToverTPSF_mask
            ngal        = np.sum(total_mask)

            sigma_e_squared = 1/2*(np.sum(e1**2)/ngal**2 + np.sum(e2**2)/ngal**2)
            mean_response   = np.mean(mcal_response[total_mask])

            sigma_gamma_squared = sigma_e_squared/mean_response**2

            weight[i, j] = 1/sigma_gamma_squared 

            weight_per_galaxy[total_mask] = 1/sigma_gamma_squared 
        

if name == '__main__':
    
    pass