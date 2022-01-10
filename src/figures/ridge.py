#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import interpolate

# Curtis et al. 2020 table
def curtis_bprp_teff(bprp):
    #Estimating effective temperature from the dereddened Gaia DR2 (Bp-Rp) color
    bprp  = np.array(bprp)
    coeff = [-416.585, 39780.0, -84190.5, 85203.9, -48225.9, 15598.5, -2694.76, 192.865]    
    teff  = np.array([np.sum([co*_bprp**i for i,co in enumerate(coeff)]) for _bprp in bprp])
    mask  = (bprp>=0.55) & (bprp<=3.25)
    teff[~mask] = np.nan
    return teff

def curtis_gyrochrone(bprp, kind):
    bprp = np.array(bprp)
    
    if kind=='kepler': #Kepler lower envelope
        bprp_min, bprp_max = 0.6, 2.1
        coeff = [36.4756, -202.718, 414.752, -395.161, 197.800, -50.0287, 5.05738]
        
    elif kind=='pleiades-ro':
        bprp_min, bprp_max = 0.6, 1.3
        coeff = [37.068, -188.02, 332.32, -235.78, 60.395]

    elif kind=='pleiades-quad':
        bprp_min, bprp_max = 0.6, 1.3
        coeff = [-8.467, 19.64, -5.438]
        
    elif kind=='praesepe':
        bprp_min, bprp_max = 0.6, 2.4
        coeff = [-330.810, 1462.48, -2569.35, 2347.13, -1171.90, 303.620, -31.9227]
        
    elif kind=='ngc6811':
        bprp_min, bprp_max = 0.65, 1.95 
        coeff = [-594.019, 2671.90, -4791.80, 4462.64, -2276.40, 603.772, -65.0830]
        
    elif kind=='ngc752':
        bprp_min, bprp_max = 1.32, 2.24
        coeff = [6.80, 5.63] 
    
    elif kind=='ngc6819+ruprecht147':
        bprp_min, bprp_max = 0.62, 2.07
        coeff = [-271.783, 932.879, -1148.51, 695.539, -210.562, 25.8119]
        
    prot  = np.array([np.sum([co*_bprp**i for i,co in enumerate(coeff)]) for _bprp in bprp])
    mask  = (bprp>=bprp_min) & (bprp<=bprp_max)
    prot[~mask] = np.nan
    
    return prot
        

#Re-casting the Curtis et al. 2020 polynomial relations in Teff
def curtis_teff_gyrochrone(teff, kind):
    
    _bprp = np.linspace(0,5,10000)
    _teff = curtis_bprp_teff(_bprp)
    _prot = curtis_gyrochrone(_bprp, kind)
    
    _ = (np.isfinite(_teff)) & (np.isfinite(_prot))
    
    # Be cognizant that using "extrapolate" means the resulting relations will be unreliable
    # outside the Teff ranges over which they were derived, but for our purposes it is effective 
    f = interpolate.interp1d(_teff[_], 
                             _prot[_], 
                             kind='cubic', 
                             fill_value='extrapolate')
    
    return f(teff)


def curtis_teff_bprp(teff):
    #Invert Teff-BpRp relation
    
    _bprp = np.linspace(0.55,3.25,10000)
    _teff = curtis_bprp_teff(_bprp)
    
    _ = (np.isfinite(_teff)) & (np.isfinite(_bprp))
    
    # Be cognizant that using "extrapolate" means the resulting relations will be unreliable
    # outside the Teff ranges over which they were derived, but for our purposes it is effective 
    f = interpolate.interp1d(_teff[_], _bprp[_], kind='cubic', fill_value='extrapolate')
    
    return f(teff)


#McQuillan et al. 2013 rotation periods for Kepler KOIs
mcq_koi = Table.read("https://cdsarc.cds.unistra.fr/ftp/J/ApJ/775/L11/table1.dat",
                readme="https://cdsarc.cds.unistra.fr/ftp/J/ApJ/775/L11/ReadMe",
                format="ascii.cds")
mcq_koi = mcq_koi.to_pandas()
mcq_koi = mcq_koi.add_prefix('mcq_')
mcq_koi.head()


#CKS catalog with auxiliary data
#Originally from Fulton & Petigura 2018, with auxiliary literature data compiled in David et al. 2021
cks = pd.read_parquet('../data/cks_merged.parquet')
# The dataframe has a row entry for each KOI, meaning individual star are represented N times
# where N is the number of KOIs detected around that star so we drop duplicates.
cks = cks.drop_duplicates(subset=['kepid'], keep='first')
cks = cks.merge(mcq_koi, how='left', left_on='kepid', right_on='mcq_KIC')
cks.head()

sns.set(font_scale=2.1, context="paper", style="ticks")

def ridge_hi(teff):
    m = (2-24)/(6500-5800)
    b = (2 - m*6500) 
    return m*teff + b

def ridge_lo(teff):
    m = (2-24)/(6500-5800)
    b = (-5 - m*6500) 
    return m*teff + b

mask = (cks['p20_cks_slogg']>4) #main sequence

# Parallelogram
xp = [6500,6500,5800,5800]
yp = [ridge_lo(6500),ridge_hi(6500),ridge_hi(5800),ridge_lo(5800)]

x = cks['cks_Teff']
ys = [cks['m13_Prot'], 
      cks['m15_Prot'], 
      cks['a18_period'], 
      cks['d21_prot']]

labels = ['McQuillan et al. 2013',
          'Mazeh et al. 2015',
          'Angus et al. 2018',
          'David et al. 2021']

sc_kws = {'cmap': "Blues_r", 
          's':25, 
          'vmin':0, 
          'vmax':10, 
          'alpha':0.75, 
          'rasterized':True, 
          'edgecolor':'k', 
          'lw':0.25}

gyro_sequences = ['pleiades-ro',
                  'praesepe',
                  'ngc6811',
                  'ngc6819+ruprecht147']

fig, ax = plt.subplots(nrows=1, ncols=len(ys), figsize=(0.8*25,0.8*6.5))

for i,y in enumerate(ys):
    sc1 = ax[i].scatter(x[mask], y[mask], c=cks['cks_age'][mask], **sc_kws)
    
    cb1 = fig.colorbar(sc1, ax=ax[i], pad=0)
    
    ax[i].set_title(labels[i])
    ax[i].set_ylim(0,45)
    ax[i].set_xlim(6500,5000)
    ax[i].set_ylabel('Rotation period [d]')
    ax[i].set_xlabel('Effective temperature [K]')
    cb1.set_label('Age [Gyr]')
           
    _teff = np.linspace(4500,6500,1000)

    for seq in gyro_sequences:
        ax[i].plot(_teff, curtis_teff_gyrochrone(_teff, kind=seq), label=seq, color='k', lw=4, alpha=0.2)
        
    #Parallelogram
    ax[i].add_patch(patches.Polygon(xy=list(zip(xp,yp)), fill=False, lw=1, color='k'))

sns.despine()    
plt.tight_layout()
plt.savefig('../figures/ridge.pdf')
plt.show()


