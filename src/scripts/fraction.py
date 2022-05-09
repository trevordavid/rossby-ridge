#!/usr/bin/env python
# coding: utf-8

import paths
import numpy as np
import pandas as pd

from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300

import seaborn as sns

from scipy import interpolate

import astropy.constants as c

sun = {"teff": 5772,
       "prot": 25.4,
       "e_prot": 25.4-24.5,
       "E_prot": 27-25.4
      }

sun["logg"] = np.log10(c.GM_sun.cgs.value/c.R_sun.cgs.value**2)

######################################################################################
#McQuillan et al. 2014
mcq = pd.read_parquet(paths.data / 'mcquillan2014_table1.parquet')
######################################################################################

######################################################################################
# LAMOST-Kepler 
lam = pd.read_parquet(paths.data / 'kepler_lamost.parquet')
print('LAMOST unique KIC targets:', len(np.unique(lam["KIC"])))
print('LAMOST unique DR2 targets:', len(np.unique(lam["DR2Name"])))

# Drop duplicate sources, keeping the one with the brighter G magnitude
lam = lam.sort_values(["KIC", "Gmag"], ascending = (True, True))
lam = lam.merge(mcq, how='left', left_on="KIC", right_on="mcq_KIC")
lam = lam.drop_duplicates(subset=['KIC'], keep='first')

lam_mask = (lam["Teff_lam"]>3000)
lam_mask = (lam["Teff_lam"]<8000)
lam_mask &= (lam["logg_lam"]>4.1)
lam_mask &= (lam["logg_lam"]<5)
lam_mask &= (abs(lam["feh_lam"])<2)
lam = lam[lam_mask]

print('LAMOST unique KIC targets:', len(np.unique(lam["KIC"])))
print('LAMOST unique DR2 targets:', len(np.unique(lam["DR2Name"])))
print('Median LAMOST Teff error:', np.median(lam["e_Teff_lam"]))
######################################################################################

######################################################################################
def convective_turnover_timescale(teff):    
    #Returns convective turnover timescale in days
    #Gunn et al. 1998 relation, from Cranmer & Saar 2011
    return 314.24*np.exp( -(teff/1952.5) - (teff/6250.)**18. ) + 0.002
    
def constant_rossby(teff, ro):
    #Return locus of rotation periods corresponding to constant Rossby number
    return ro * convective_turnover_timescale(teff)
######################################################################################


dist = abs(lam["Prot"] - constant_rossby(lam["Teff_lam"], 1.3))
frac_dist = abs(lam["Prot"] - constant_rossby(lam["Teff_lam"], 1.3))/constant_rossby(lam["Teff_lam"], 1.3)
lam_ridge = (frac_dist<0.05) & (lam["Teff_lam"]>5500) & (lam["Teff_lam"]<6500) & (lam["logg_lam"]>4) & (lam["logg_lam"]<4.75)

sns.set(font_scale=1.5, context="paper", style="ticks", palette="Blues")

xeps = 5
yeps = 1

x = np.arange(5000,6500+xeps,xeps)
relfrac_lp = np.zeros(len(x))
relfrac_sp = np.zeros(len(x))

for i,xcen in enumerate(x):
    
    ycen_lp = constant_rossby(xcen, 1.3)
    ycen_sp = constant_rossby(xcen, 0.5)
    
    arg_slice  = (abs(lam["Teff_lam"]-xcen)<xeps)
    arg_lp = arg_slice & (abs(lam["Prot"]-ycen_lp)<yeps)
    arg_sp = arg_slice & (abs(lam["Prot"]-ycen_sp)<yeps)
    
    relfrac_lp[i] = len(lam["Prot"][arg_lp])/len(lam["Prot"][arg_slice])
    relfrac_sp[i] = len(lam["Prot"][arg_sp])/len(lam["Prot"][arg_slice])
    
plt.plot(x, relfrac_lp, 'C2', label='Long-period pile-up', lw=3, alpha=0.5)
plt.plot(x, relfrac_sp, 'C5', label='Short-period pile-up', lw=3, alpha=0.5)
plt.axvline(sun["teff"], color='k', ls='--', label='Sun')
plt.xlim(6600,4900)
plt.ylabel("Relative fraction of pile-up stars")
plt.xlabel("Effective temperature [K]")
plt.legend()
sns.despine()
plt.savefig(paths.figures / 'fraction.pdf')
