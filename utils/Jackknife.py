#############################################################
# Set of utils for doing jackknife for shear tests
#############################################################

import treecorr
import numpy as np
from tqdm import tqdm


class Jackknife(object):
    '''
    Class for Jackknifing any measurement of interest to us
    '''
    
    def __init__(self, ra, dec, Npatch):
        '''
        Input sky coords and organize catalog into patches
        using treecorr.

        ra: 
                right ascension, in degrees. 
        dec: 
                declination, in degrees.
        Npatch:
                The number of patches to split into


        returns

            np.array : the index of patch each catalog entry is assigned to
        '''
        
        self.patch_ind = treecorr.Catalog(ra = ra, dec = dec, ra_units = 'deg', dec_units = 'deg', npatch = Npatch).patch
        self.Npatch    = Npatch
    
    
    def jackknife_sample(self, i, patches):
        '''
        Returns array indices of elements to use to jackknife sample, i

        i: 
                index
        patch_ind: 
                declination, in degrees.
        Npatch:
                The number of patches to split into
        '''

        return np.where(patches != i)[0]

    
    def go(self, my_function, **kwargs):
        '''
        Run jackknife with your statistic
        
        my_function:
                A function that takes in all of the quantities in provided in kwargs
                and returns an array of the results.
        
        **kwargs:
                Dictionary containing all keyword arguments passed to go.
                The keywords should be the same names that ``my_function`` is expecting.
                
                
        ------------
        Output
        ------------
        
        np.array:
                The mean value of the statistic averaged across all jackknifes
        
        np.array:
                The covariance of the statistic
                
        list:
                The individual realizations of the statistic measured on
                each jackknife realization
        '''

        
        Output = []

        for i in tqdm(range(self.Npatch), desc = 'Jackknifing %s' % my_function.__name__):

            inds = self.jackknife_sample(i, self.patch_ind)
            new_kwargs = {k : kwargs[k][inds] for k in kwargs.keys()}

            out  = my_function(**new_kwargs)
            Output.append(out)

        mean = np.mean(Output, axis = 0)
        cov  = self.jackknife_cov(np.array(Output))
        
        return mean, cov, Output

    
    def jackknife_cov(self, X):
        '''
        Compute the covariance of a given set of jackknife measurements

        X:
            np.array of dimension (N_jackknife, N_datavector)

        returns

            np.array : covariance matrix of shape (N_datavector, N_datavector)
        '''

        N_jackknife = X.shape[0]

        cov = np.cov(X.T) * (N_jackknife - 1)

        return cov
    
    
if __name__ == '__main__':
    
    #Make some example data
    
    Ngal = 1_000_000
    ra, dec = np.random.uniform(30, 180, Ngal), np.random.uniform(-60, 30, Ngal)
    e1, e2  = np.random.uniform(-1, 1, Ngal), np.random.uniform(-1, 1, Ngal)
    
    
    knife_it = Jackknife(ra, dec, 100)
    
    
    def bin_shape_variance(e1, e2):
        X = 1/2 * (e1**2 + e2**2)
        return np.histogram(X, bins = np.linspace(0, 1, 10))[0]
    
    mean, cov, raw_output = knife_it.go(bin_shape_variance, e1 = e1, e2 = e2)
    
    print("MEAN:", mean)
    print("COV :", cov)
    
    
    def bin_shape_average(e1, e2):
        X = 1/2 * (e1 + e2)
        return np.histogram(X, bins = np.linspace(0, 1, 10))[0]
    
    mean, cov, raw_output = knife_it.go(bin_shape_average, e1 = e1, e2 = e2)
    
    print("MEAN:", mean)
    print("COV :", cov)
    
    