#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as c
from astropy.table import Table

from scipy import interpolate

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300

import seaborn as sns


# ### Hall et al. 2021 asteroseismic sample
hall = Table.read("https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/table1.dat",
                  readme="https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/ReadMe",
                  format="ascii.cds")
#hall.info()


# ### CKS sample
#McQuillan et al. 2013
mcq_koi = Table.read("https://cdsarc.cds.unistra.fr/ftp/J/ApJ/775/L11/table1.dat",
                readme="https://cdsarc.cds.unistra.fr/ftp/J/ApJ/775/L11/ReadMe",
                format="ascii.cds")
mcq_koi = mcq_koi.to_pandas()
mcq_koi = mcq_koi.add_prefix('mcq_')
#mcq_koi.head()

cks = pd.read_parquet('../data/cks_merged.parquet')
# The dataframe has a row entry for each KOI, meaning individual star are represented N times
# where N is the number of KOIs detected around that star so we drop duplicates.
print(np.shape(cks))
cks = cks.drop_duplicates(subset=['kepid'], keep='first')

cks = cks.merge(mcq_koi, how='left', left_on='kepid', right_on='mcq_KIC')
print(np.shape(cks))
cks.head()


# ### LAMOST-Kepler sample
lam = pd.read_csv('../data/kepler_lamost.csv')

#Drop duplicates
lam = lam.drop_duplicates(subset=['KIC'], keep='first')
lam.head()

#ik = (lam['Teff_lam']>0)
#ik &= (lam['Gmag']<15)

#ik &= (lam['l20_Rvar']/lam['rrmscdpp03p0'] > 10)
#ik &= (lam['logg_lam']-3*lam['e_logg_lam']>4)
#ik &= (lam['duplicated_source']==False)
#ik &= (abs(lam["Gmag"]-lam["phot_g_mean_mag"])<1)
#ik &= (abs(lam["Gmag"]-lam["kepmag"])<1)
#ik &= (abs(lam["Teff_lam"]-lam["teff_val"])<1000)
#ik &= (lam['parallax_over_error']>50)
#ik &= (lam["vim_r"]<0.5)
#ik &= (lam["ruwe"]<1.2)


# ### Comparison with Hall et al. 2021 asteroseismic sample
sns.set(font_scale=1.2, context="paper", style="ticks")

mpl.rcParams["legend.markerscale"] = 2
titles = ["California–Kepler Survey","LAMOST–McQuillan"]
ebar_kws = {"fmt":"o", "lw":1, "ms": 4, "ecolor":"orange", "color":"orange", "alpha":1, "mec":"grey", "mew":0.5}

sc_kws = {"marker":".", "color":"C0", "rasterized":True, "alpha":1}

fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(4.5,6.5))

ax[0].scatter(cks["p20_cks_steff"], cks["d21_prot"], **sc_kws, label=r"California–Kepler Survey", zorder=999, s=4)
ax[1].scatter(lam['Teff_lam']+41, lam['Prot'], **sc_kws, label=r"LAMOST–McQuillan", s=0.5)
#Note: the 41 K offset is derived in the appendix

for i in range(2):
    arg = hall["Type"] == "MS"
    ax[i].errorbar(hall["Teff"][arg], hall["P"][arg], 
                   xerr=hall["e_Teff"][arg], 
                   yerr=[hall["e_P"][arg], hall["E_P"][arg]], 
                   label="Hall et al. 2021\nasteroseismic sample",
                   **ebar_kws)
    ax[i].set_xlim(6500,5000)
    ax[i].set_xlabel("Effective temperature [K]")
    ax[i].set_ylabel("Rotation period [d]")
    #ax[i].set_title(titles[i])
    ax[i].legend(prop={"size":9})
    
sns.despine()    
plt.tight_layout()
plt.savefig('../figures/asteroseismic.pdf')
plt.show()
