#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import astropy.constants as c
from astropy.table import Table
from astropy.table import join

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300

import seaborn as sns

sun = {"teff": 5772,
       "prot": 25.4,
       "e_prot": 25.4-24.5,
       "E_prot": 36-25.4
      }

sun["logg"] = np.log10(c.GM_sun.cgs.value/c.R_sun.cgs.value**2)

######################################################################################
#McQuillan et al. 2013
mcq_koi = Table.read("https://cdsarc.cds.unistra.fr/ftp/J/ApJ/775/L11/table1.dat",
                readme="https://cdsarc.cds.unistra.fr/ftp/J/ApJ/775/L11/ReadMe",
                format="ascii.cds")
mcq_koi = mcq_koi.to_pandas()
mcq_koi = mcq_koi.add_prefix('mcq_')

#McQuillan et al. 2014
# mcq = Table.read('../data/mcquillan2014/table1.dat',
#                 readme='../data/mcquillan2014/ReadMe',
#                 format='ascii.cds')
# mcq = mcq.to_pandas()
# mcq = mcq.add_prefix('mcq_')
mcq = pd.read_parquet('../data/mcquillan2014_table1.parquet')
######################################################################################


######################################################################################
# California-Kepler Survey (Fulton & Petigura 2018)
# This data table has been augmented with data from other surveys (see David et al. 2021)
cks = pd.read_parquet('../data/cks_merged.parquet')
# The dataframe has a row entry for each KOI, meaning individual star are represented N times
# where N is the number of KOIs detected around that star so we drop duplicates.
cks = cks.drop_duplicates(subset=['kepid'], keep='first')
cks = cks.merge(mcq_koi, how='left', left_on='kepid', right_on='mcq_KIC')
######################################################################################


######################################################################################
# LAMOST-Kepler 
lam = pd.read_csv('../data/kepler_lamost.csv')
print('LAMOST unique KIC targets:', len(np.unique(lam["KIC"])))
print('LAMOST unique DR2 targets:', len(np.unique(lam["DR2Name"])))

# Drop duplicate sources, keeping the one with the brighter G magnitude
lam = lam.sort_values(["KIC", "Gmag"], ascending = (True, True))
lam = lam.merge(mcq, how='left', left_on="KIC", right_on="mcq_KIC")
lam = lam.drop_duplicates(subset=['KIC'], keep='first')

lam_mask = (lam["Teff_lam"]>3000)
lam_mask = (lam["Teff_lam"]<8000)
lam_mask &= (lam["logg_lam"]>3)
lam_mask &= (lam["logg_lam"]<5)
lam_mask &= (abs(lam["feh_lam"])<2)
lam = lam[lam_mask]

print('LAMOST unique KIC targets:', len(np.unique(lam["KIC"])))
print('LAMOST unique DR2 targets:', len(np.unique(lam["DR2Name"])))
print('Median LAMOST Teff error:', np.median(lam["e_Teff_lam"]))
######################################################################################

######################################################################################
# APOGEE-Kepler (from Adrian Price-Whelan, using APOGEE DR16)
apo = Table.read('../data/kepler_apogee_dr16.fits')
names = [name for name in apo.colnames if len(apo[name].shape) <= 1]
apo  = apo[names].to_pandas()
apok = apo.merge(mcq, how='inner', left_on='kepid', right_on='mcq_KIC')
######################################################################################

sns.set(font_scale=1.4, context="paper", style="ticks")

sc_kws = {"marker":".", "alpha": 1, "rasterized":True, "color":"C0"}

titles=['KIC +\nMcQuillan et al. 2014', 
        'LAMOST DR5 +\nMcQuillan et al. 2014', 
        'APOGEE DR16 +\nMcQuillan et al. 2014', 
        'CKS–Gaia +\nMcQuillan et al. 2013']

fig,ax = plt.subplots(nrows=1,ncols=4, figsize=(14,3))

ax[0].scatter(mcq["mcq_Teff"], mcq["mcq_Prot"], **sc_kws, s=0.01)
ax[1].scatter(lam["Teff_lam"], lam["mcq_Prot"], **sc_kws, s=0.1)
ax[2].scatter(apok["TEFF"], apok["mcq_Prot"], **sc_kws, s=0.5)
ax[3].scatter(cks["cks_Teff"], cks["mcq_Prot"], **sc_kws, s=2)

for i in range(4):
    ax[i].set_title(titles[i])
    ax[i].set_xlabel("Effective temperature [K]")        
    ax[i].set_xlim(7250,3250)
    ax[i].set_ylim(-2,60)
    ax[i].plot(sun["teff"], sun["prot"], 'o', color='C1')
    ax[i].errorbar(sun["teff"], sun["prot"], yerr=np.vstack([sun["e_prot"], sun["E_prot"]]), fmt="o", 
                   color="C1", mec="white", ms=6)
    
    
    if i>0:
        ax[i].set_yticklabels([])
    

ax[0].set_ylabel("Rotation period [d]")    
sns.despine()
plt.subplots_adjust(wspace=0)
plt.savefig('../figures/surveys.pdf')
plt.savefig('../static/surveys.png')
