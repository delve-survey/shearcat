#!/usr/bin/python

#Author: Juan de Vicente
#17/01/2015
#photoz.py
#v 1.0
#Usage: python photoz.py

#Program to test the functions of nf (Neigborhood Fit) module

import math
import numpy as np
import sys
import fnmatch

import sys
import dnf
import astropy
from astropy.table import Table


#parameters
nfilters=4

#####reading training file ######################

GALAXY=Table.read(sys.argv[1]) 
#GALAXY=GALAXY[:1000]
Ngalaxies=len(GALAXY)
print('Ngalaxies=',Ngalaxies)

G=np.zeros((Ngalaxies,nfilters),dtype='double')
Gerr=np.zeros((Ngalaxies,nfilters),dtype='double')


G[:,0]=GALAXY['mag_g']
G[:,1]=GALAXY['mag_r']
G[:,2]=GALAXY['mag_i']
G[:,3]=GALAXY['mag_z']
#G[:,4]=GALAXY['mag_Y']

Gerr[:,0]=GALAXY['magerr_g']
Gerr[:,1]=GALAXY['magerr_r']
Gerr[:,2]=GALAXY['magerr_i']
Gerr[:,3]=GALAXY['magerr_z']
#Gerr[:,4]=GALAXY['magerr_Y']


Ntrain=Ngalaxies
TRAIN=GALAXY
TRAIN['Z']=TRAIN['redshift']
T=G
Terr=Gerr

#read valid
#GALAXY=Table.read(sys.argv[2]) 
GALAXY=Table.read(sys.argv[2]) 
print('GALAXY=',GALAXY)
Ngalaxies=len(GALAXY)
print('Ngalaxies=',Ngalaxies)

G=np.zeros((Ngalaxies,nfilters),dtype='double')
Gerr=np.zeros((Ngalaxies,nfilters),dtype='double')

G[:,0]=GALAXY['mag_g']
G[:,1]=GALAXY['mag_r']
G[:,2]=GALAXY['mag_i']
G[:,3]=GALAXY['mag_z']
#G[:,4]=GALAXY['mag_Y']

Gerr[:,0]=GALAXY['magerr_g']
Gerr[:,1]=GALAXY['magerr_r']
Gerr[:,2]=GALAXY['magerr_i']
Gerr[:,3]=GALAXY['magerr_z']
#Gerr[:,4]=GALAXY['magerr_Y']

Nvalid=Ngalaxies
VALID=GALAXY #[:100000]
V=G
Verr=Gerr
#################################### 
      


#bins
#start=0.0
#stop=0.8
#step=0.1

#start
#step=0.066

start=0.0
stop=1.6
step=0.01

#start=0.0
#stop=2.0
#step=0.1

#start=0.0
#stop=0.9
#step=0.1

#start=0.1
#stop=0.7
#step=0.0375

#start=np.double(raw_input('start:'))
#stop=np.double(raw_input('stop:'))
#step=np.double(raw_input('step:'))

zbins=np.arange(start,stop,step)
nbins=len(zbins)-1 
bincenter=(np.double(zbins[1:])+np.double(zbins[:-1]))/2.0

#reponer
#zbins = np.linspace(0, 2.0, 50) #201)
#bincenter = (zbins[0:-1] + zbins[1:])/2.0
#nbins=len(zbins)-1

binning = zbins  #np.linspace(0, 2.0, 201)
bin_centers = (binning[0:-1] + binning[1:])/2.0


algorithm=sys.argv[3]
#algorithm=raw_input('\nenf\ndnf\nanf\nEnter an option:')

#names=('z_photo','z1','zerr','zerrabs','zerr_e','mode_z','mean_z','sample_z','std_z','SNmean','SNmax'

#PHOTOZ CALL
z_photo,zerr_e,photozerr_param,photozerr_fit,Vpdf,z1,nneighbors,de1,d1,id1,C=dnf.dnf(T,TRAIN['Z'],V,Verr,zbins,pdf=False,Nneighbors=80,bound=False,radius=2,magflux='mag',metric="DNF",coeff=True) 

print("mean Nneighbors=",np.mean(nneighbors))


#selection
print('Nvalid before=',Nvalid)
#selection=zerr_e!=99.0
#selection=VALID['closestDistance']<=R
#selection=VALID['nneighbors']>30
#z_photo=z_photo[selection]
#z1=z1[selection]
#zerr_e=zerr_e[selection]
#VALID=VALID[selection]
#Vpdf=Vpdf[selection]
#NvalidBefore=Nvalid

#Nvalid=VALID.shape[0]
#print 'Nvalid after=',Nvalid

#print 'percentage=',Nvalid*100.0/NvalidBefore
 


#SAVE RESULTS
#*********point prediction file**********
#f.writeto('test.fits')
#create the test fits files 
from astropy.table import Table
#d = {} #dictionary
d=VALID
#COADD_OBJECTS_ID MAG_DETMODEL_I     WEIGHTS          MEAN_Z          MODE_Z         Z_SPEC          Z_MC 
#d['REDSHIFT'] = VALID['REDSHIFT']
#d['Z_SPEC'] = VALID['REDSHIFT']
#d['COADD_OBJECT_ID'] = VALID['COADD_OBJECT_ID']
#d['RA'] = VALID['RA']
#d['DEC'] = VALID['DEC']
#d['TILENAME']=VALID['TILENAME']
#d['WEIGHTS'] = VALID['weights_valid']
##d['weights_valid'] = 0.0 #['WEIGHTS']
#d['weights_valid'] = VALID['weights_valid'] #'WEIGHTS'
#a=np.ones(Nvalid,dtype='double')

#d['WEIGHTS'] =a
#d['weights_valid'] =a
#d['MODE_Z']= z_photo
#d['MEAN_Z']= z_photo
#d['MEDIAN_Z']= z_photo
#d['Z_MC']= z1
#d['MAG_DETMODEL_I']=VALID['MAG_AUTO_I']

#d['flux_g']=mag[:][0]
#d['flux_r']=mag[:][1]
#d['flux_i']=mag[:][2]
#d['flux_z']=mag[:][3]

d['DNF_Z']=z_photo #+algorithm.lower()]=z_photo
d['DNF_ZN']=z1 #+algorithm.lower()]=z1
d['DNF_ZSIGMA']=zerr_e
d['photozerr_param']=photozerr_param
d['photozerr_fit']=photozerr_fit
d['d1']=d1
#d['Z']=VALID['Z']
d['de1']=de1
d['id1']=id1
#d['M']=V
#d['M1']=T[id1]
#d['C']=C
# d['Vpdf']=Vpdf

#d['Vpdf_start']=str(start)
#d['Vpdf_stop']=str(stop)
#d['Vpdf_step']=str(step)

#d['numberOfNeighgors']=nneighbors
#d['closestDistance']=closestDistance
#d['zerr_'+algorithm.lower()]=zerr_e
#d['nneighbors_'+algorithm.lower()]=nneighbors
#d['closestDistance_'+algorithm.lower()]=closestDistance
#print 'd=',d
#fit = Table(d)
#fit.write('test1wlenf/jvicente'+'_'+algorithm+'_'+testfile, format='fits',overwrite=True)
d.write(sys.argv[4], format='fits',overwrite=True)


#raw_input()


sys.exit()




#####Signal to noise analysis
Msnm=np.zeros(Nvalid,dtype='double')
Msnstd=np.zeros(Nvalid,dtype='double')
Msnsum=np.zeros(Nvalid,dtype='double')
Msnmax=np.zeros(Nvalid,dtype='double')
Msnmin=np.zeros(Nvalid,dtype='double')
Msnmod=np.zeros(Nvalid,dtype='double')

####z1 results
#photoz error
zerr1=z1-VALID['REDSHIFT']
zerrabs1=np.abs(z1-VALID['REDSHIFT'])  
print('z1err results')
print('mean=',zerr1.mean())
print('median=',np.median(zerr1))
print('std=',zerr1.std())
print('mad=',zerrabs1.mean())
print('biasNorm=',(zerr1/zerr_e).mean())
print('sigmaNorm=',(err1/zerr_e).std())

zerr1Sort=np.sort(zerrabs1)
sigma68=zerr1Sort[Nvalid*68/100]
print('sigma68=',sigma68)
aux=np.where(np.sqrt(z_true_hist)==0.0,0.0,(z_true_hist-z_photo_hist)/np.sqrt(z_true_hist))
print('Npoisson_Nz=',np.linalg.norm(aux)/np.sqrt(nbins))
print('kS=',ks_2samp(z_true_hist,z_photo_hist)[0])

aux=np.where(np.sqrt(z_true_hist)==0.0,0.0,(z_true_hist-z_1_hist)/np.sqrt(z_true_hist))
print('Npoisson_Nz_1=',np.linalg.norm(aux)/np.sqrt(nbins))
print('kS_1=',ks_2samp(z_true_hist,z_1_hist)[0])

#print 'Npoisson_Nz=',(np.linalg.norm((z_true_hist-z_photo_hist)/np.sqrt(z_true_hist)))/np.sqrt(nbins)
print('Chi2_Nz=',np.linalg.norm((z_true_hist-z_photo_hist)/np.sqrt(z_photo_hist)))  #/np.sqrt(nbins)

#END
sys.exit()
