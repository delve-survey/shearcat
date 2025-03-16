#!/usr/bin/python
"""
nf module: contains two alternative functions to compute the photoz of a photometric sample of galaxies.
enf: Euclidean Neighborhood Fit (faster) 
dnf: Directional Neighborhood Fit (better)

The following python libraries are used in the code:
import math
import numpy
from sklearn import neighbors  
"""
__author__ = "Juan de Vicente"
__copyright__ = "Copyright 2015, Juan de Vicente"
__version__ = "4.0.3"
__email__= "juan.vicente@ciemat.es"

import math
import numpy as np
from sklearn import neighbors



def dnf(T,z,V,Verr,zbins,pdf=True,bound=True,radius=2.0,Nneighbors=80,magflux='flux',metric='DNF',coeff=True):
    """
    def dnf(T,z,V,Verr,zbins,pdf=True,bound=True,radius=2.0,Nneighbors=80,magflux='flux',metric='DNF')
    
    Computes the photo-z by Directional Neighborhood Fit (Copyright (C) 2015, Juan de Vicente)
  
    Input parameters:
      T: 2-dimensional array with magnitudes of the training sample
      z: 1-dimensional array with the spectroscopic redshift of the training sample
      V: 2-dimensional array with magnitudes of the photometric sample
      Verr: 2-dimensional array with magnitudes errors of the photometric sample
      zbins: 1-dimensional numpy array with redshift bins for photo-z PDFs
      pdf: True for pdf computation
      bound: True to ensure photometric redshifts remain inside the training redshift range.
      radius: Euclidean-radius for euclidean neighbors preselection to speed up and avoid outliers.
      Galaxies without neighbors inside this radius are tagged with photoz_err=99.0 and should be removed from statistical analysis.
      Nneighbors: Number of neighbors to construct the photo-z hyperplane predictor (number of neighbors for the fit)
      magflux: 'mag' | 'flux'
      metric: 'ENF' | 'ANF' | 'DNF' (euclidean|angular|directional)
      coeff: True for returning the fitting coeffients 
    Return:
      photoz: 1-dimesional dnf-photoz array for the photometric sample
      photozerr: 1-dimensional photoz error estimation array. Takes the value 99.0 for galaxies with unreliable photo-z
      photozerr_param: photo-z uncertainty due to photometry uncertainty
      photozerr_fit: photo-z uncertainty coming from fit residual
      Vpdf: 2-dimensional photo-z PDFs array when pdf==1, 0 when pdf==0
      z1: 1-dimesional photo-z array to be used for histogramming N(z). When computing n(z) per bin, use dnf-photoz for galaxy classification in bins and z1 for n(z) histogramming.
      nneighbors: 1-dimensional array with the number of neighbors used in the photo-z estimation for each galaxy
      de1: 1-dimensional array with the Euclidean magnitude distance to the nearest neighbor for each galaxy
      d1: 1-dimensional array with the metric-distance to the nearest neighbor for each galaxy
      id1: 1-dimensional array with the position of the nearest-neighbor for each galaxy (with metric-distance)
      C: C=fit-coeficients when coeff==True, otherwise C=0
    """      
    
    nfilters=T.shape[1]
    Nvalid=V.shape[0]
    Ntrain=T.shape[0]
    
    if Ntrain>4000: #2000
        Nneighbors_presel=4000 #2000
    else:
        Nneighbors_presel=Ntrain
        
    #neighbor preselection within radius mag-euclidean metric
    clf=neighbors.KNeighborsRegressor(n_neighbors=Nneighbors_presel)
    clf.fit(T,z)  #multimagnitude-redshift association from the training sample
    #photoz=clf.predict(V)
    Vdistances,Vneighbors= clf.kneighbors(V,n_neighbors=Nneighbors_presel)  #neighbors computation
    de1=Vdistances[:,0]  #euclidean nearest-neighbor euclidean distance
    d1=Vdistances[:,0]  #nearest-neighbor metric distance (to be properly overwritten latter)
    Vclosest=Vneighbors[:,0]
    id1=Vclosest
    

    #In case of giving fluxes compute closest distance in magnitude
    if magflux=='flux':
        for i in range(Nvalid):
            magV=-2.5*np.log10(V[i])
            magT=-2.5*np.log10(T[Vclosest[i]])
            diff=magV-magT
            dmag=np.sqrt(np.inner(diff,diff))
            de1[i]=dmag
        
   
    #output declaration
    photoz=np.zeros(Nvalid,dtype='double')
    z1=np.zeros(Nvalid,dtype='double')
    photozerr=np.zeros(Nvalid,dtype='double')
    photozerr_param=np.zeros(Nvalid,dtype='double')
    photozerr_fit=np.zeros(Nvalid,dtype='double')
    nneighbors=np.zeros(Nvalid,dtype='double')
      
    #auxiliary variable declaration
    pescalar=np.zeros(Ntrain,dtype='double')
    D2=np.zeros(Ntrain,dtype='double')
    Tnorm=np.zeros(Ntrain,dtype='double')
    Tnorm2=np.zeros(Ntrain,dtype='double')    
    #max and min training photo-zs
    maxz=np.max(z)
    minz=np.min(z)

    if coeff==True:
         C=np.zeros((Nvalid,nfilters+1),dtype='double')
    else:
         C=0

    ########pdf bins##########
    nbins=len(zbins)-1 
    bincenter=(np.double(zbins[1:])+np.double(zbins[:-1]))/2.0
    if pdf==True:
     Vpdf=np.zeros((Nvalid,nbins),dtype='double')
    else:
        Vpdf=0

    #training flux/mag norm pre-calculation
    for t,i in zip(T,range(Ntrain)):
     Tnorm[i]=np.linalg.norm(t)
     Tnorm2[i]=np.inner(t,t)

    #for offset of the fit
    Te=np.ones((Ntrain,nfilters+1),dtype='double')  
    Te[:,:-1]=T
    Ve=np.ones((Nvalid,nfilters+1),dtype='double')  
    Ve[:,:-1]=V

    #to computed neighbor pre-selection within mag radius in case of fluxes  
    ratiomax=np.power(10.0,radius/2.5)
    
    #photo-z computation
    for i in range(0,Nvalid):
        #neighbors pre-selection within mag radius
        if magflux=='mag':
              selection=Vdistances[i]<radius
        elif magflux=='flux':
              selection=np.ones(Nneighbors_presel,dtype='bool') 
              for j in range(0,nfilters):
                  ratio1=V[i][j]/T[Vneighbors[i],j]
                  ratio2=T[Vneighbors[i],j]/V[i][j]
                  selectionaux=np.logical_and(ratio1<ratiomax,ratio2<ratiomax) 
                  selection=np.logical_and(selection,selectionaux)

        Vneighbo=Vneighbors[i][selection]
        Vdistanc=Vdistances[i][selection]
        
        Eneighbors=Vneighbo.size #euclidean neighbors within mag radius
        if Eneighbors==0:  #probably bad photo-zs
            nneighbors[i]=0
            photozerr[i]=99.0
            photozerr_param[i]=99.0
            photozerr_fit[i]=99.0
            photoz[i]=z[Vclosest[i]]
            continue

        #declaration of auxiliary array to store neighbors features during photo-z computation 
        NEIGHBORS=np.zeros(Eneighbors,dtype=[('pos','i4'),('distance','f8'),('z_true','f8')])
        #copy of euclidean preselection previously computed  
        NEIGHBORS['z_true']=z[Vneighbo] #photo-z
        NEIGHBORS[:]['pos']=Vneighbo 
        Ts=T[Vneighbo] #flux/mag  
        
        if metric=='ENF':
             D=V[i]-Ts
             Dsquare=D*D
             D2=np.sum(Dsquare,axis=1)
             NEIGHBORS['distance']=D2
        elif metric=='ANF':
             Tsnorm=Tnorm[Vneighbo] 
             Vnorm=np.linalg.norm(V[i])
             pescalar=np.inner(V[i],Ts)
             normalization=Vnorm*Tsnorm
             NIP=pescalar/normalization
             alpha2=1-NIP*NIP
             NEIGHBORS['distance']=alpha2
        elif metric=='DNF':  #ENF*ANF
             D=V[i]-Ts
             Dsquare=D*D
             D2=np.sum(Dsquare,axis=1)
             

             Tsnorm=Tnorm[Vneighbo] 
             Vnorm=np.linalg.norm(V[i])
             pescalar=np.inner(V[i],Ts)
             normalization=Vnorm*Tsnorm
             NIP=pescalar/normalization
             alpha2=1-NIP*NIP

             D2norm=D2/(Vnorm*Vnorm) #normalized distance to do it more interpretable
             NEIGHBORS['distance']=alpha2*D2norm 
     
        NEIGHBORSsort=np.sort(NEIGHBORS,order='distance')
        z1[i]=NEIGHBORSsort[0]['z_true']
        d1[i]=NEIGHBORSsort[0]['distance']
        id1[i]=NEIGHBORSsort[0]['pos']

        #if the galaxy is found in the training sample
        if NEIGHBORSsort[0]['distance']==0.0:
            #photoz[i]=NEIGHBORSsort[0]['z_true']
            if Eneighbors>1:
                z1[i]=NEIGHBORSsort[1]['z_true']
            #if nneighbors[i]==0:
            #   photozerr[i]=0.001
            #    photozerr_param[i]=0.001
            #    photozerr_fit[i]=0.0
            #else:
            #    photozerr_fit[i]=NEIGHBORSsort['z_true'].std()
            #    photozerr_param[i]=0.0
            #    photozerr[i]=photozerr_param[i]

            if pdf==True:
                zdist=photoz[i] #-0.01 #-residuals  #for p
                hist=np.double(np.histogram(zdist,zbins)[0])
                Vpdf[i]=hist/np.sum(hist)
            #continue
        
        #limiting the number of neighbors to Nneighbors parameter
        if Eneighbors>Nneighbors:
                NEIGHBORSsort=NEIGHBORSsort[0:Nneighbors]  #from 1 in case to exclude the own galaxy
                neigh=Nneighbors
        else:
                neigh=Eneighbors
        
        nneighbors[i]=neigh
        
            
        #nearest neighbor photo-z computation when few neighbors are found (fitting is not a good option)
        if neigh<10: #<30 
            photoz[i]=np.inner(NEIGHBORSsort['z_true'],1.0/NEIGHBORSsort['distance'])/np.sum(1.0/NEIGHBORSsort['distance']) #weighted by distance
            if neigh==1:
                photozerr_param[i]=0.1
                photozerr[i]=0.1
            else:
                #photozerr[i]=np.std(NEIGHBORSsort['z_true'])
                photozerr_fit[i]=np.sqrt(np.inner((NEIGHBORSsort['z_true']-photoz[i])**2,1.0/NEIGHBORSsort['distance'])/np.sum(1.0/NEIGHBORSsort['distance']))  #weighted by distance
                photozerr[i]=photozerr_fit[i]
            if pdf==True:
                        if photozerr[i]==0:
                            s=1
                        else:
                            s=photozerr[i]
                        zdist=np.random.normal(photoz[i],s,neigh)
                        #zdist=NEIGHBORSsort['z_true']
                        hist=np.double(np.histogram(zdist,zbins)[0])
                        sumhist=np.sum(hist)
                        if sumhist==0.0:
                            Vpdf[i][:]=0.0
                        else:
                            Vpdf[i]=hist/sumhist 
            continue

    
        #Fitting when large number of neighbors exists. Removing outliers by several iterations  
        fititerations=4
        for h in range(0,fititerations):
            A=Te[NEIGHBORSsort['pos']]  
            B=z[NEIGHBORSsort['pos']]
            X=np.linalg.lstsq(A,B)
            residuals=B-np.dot(A,X[0])
    
        
            if h==0:  #PDFs computation
                photoz[i]=np.inner(X[0],Ve[i])
            
                if pdf==True:
                    zdist=photoz[i]+residuals
                    #zdist=NEIGHBORSsort['z_true']
                    hist=np.double(np.histogram(zdist,zbins)[0])
                    sumhist=np.sum(hist)
                    if sumhist==0.0:
                        Vpdf[i][:]=0.0
                    else:
                        Vpdf[i]=hist/sumhist 
                    
        
            #outlayers are removed after each iteration 
            absresiduals=np.abs(residuals)      
            sigma3=3.0*np.mean(absresiduals)
            selection=(absresiduals<sigma3)
            #NEIGHBORSsort=NEIGHBORSsort[selection]
            
            
            nsel=np.sum(selection)
            nneighbors[i]=nsel
            if nsel>10:
                NEIGHBORSsort=NEIGHBORSsort[selection]
            else:
                break
            
        C[i]=X[0]
        photoz[i]=np.inner(X[0],Ve[i])
        neig=NEIGHBORSsort.shape[0]
        if X[1].size!=0:               
            photozerr_param[i]=np.sqrt(np.inner(np.abs(X[0][:-1])*Verr[i],np.abs(X[0][:-1])*Verr[i]))
            photozerr_fit[i]=np.sqrt(X[1]/neig)
        else:
            photozerr_fit[i]=0.01

        #photoz bound
        if bound==True:
            if photoz[i]< minz or photoz[i]>maxz:
               photozerr_fit[i]+=np.abs(photoz[i]-NEIGHBORSsort[0]['z_true'])
               photoz[i]=NEIGHBORSsort[0]['z_true']
        
        
                    

        percent=np.double(100*i)/Nvalid
        
        if i % 1000 ==1:
            print('progress: ',percent,'%')

    photozerr=np.sqrt(photozerr_param**2+photozerr_fit**2)
    
    return photoz,photozerr,photozerr_param,photozerr_fit,Vpdf,z1,nneighbors,de1,d1,id1,C



