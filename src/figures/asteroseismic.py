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


# ### CKS sample
#McQuillan et al. 2013
mcq_koi = Table.read("https://cdsarc.cds.unistra.fr/ftp/J/ApJ/775/L11/table1.dat",
                readme="https://cdsarc.cds.unistra.fr/ftp/J/ApJ/775/L11/ReadMe",
                format="ascii.cds")
mcq_koi = mcq_koi.to_pandas()
mcq_koi = mcq_koi.add_prefix('mcq_')

cks = pd.read_parquet('../data/cks_merged.parquet')
# The dataframe has a row entry for each KOI, meaning individual star are represented N times
# where N is the number of KOIs detected around that star so we drop duplicates.
cks = cks.drop_duplicates(subset=['kepid'], keep='first')
cks = cks.merge(mcq_koi, how='left', left_on='kepid', right_on='mcq_KIC')

# ### LAMOST-Kepler sample
lam = pd.read_csv('../data/kepler_lamost.csv')

#Drop duplicates
lam = lam.drop_duplicates(subset=['KIC'], keep='first')

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

sc_kws = {"marker":".", "color":"C0", "rasterized":True, "alpha":1}

fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=2,figsize=(2*4.5,6.5))

ax1[0].scatter(cks["p20_cks_steff"], cks["d21_prot"], **sc_kws, label=r"California–Kepler Survey", zorder=999, s=4)
ax1[1].scatter(lam['Teff_lam']+41, lam['Prot'], **sc_kws, label=r"LAMOST–McQuillan", s=0.5)
#Note: the 41 K offset is derived in the appendix

ax2[0].scatter(cks["p20_cks_steff"], cks["d21_prot"], **sc_kws, label=r"California–Kepler Survey", zorder=999, s=4)
ax2[1].scatter(lam['Teff_lam'], lam['Prot'], **sc_kws, label=r"LAMOST–McQuillan", s=0.5)


ebar_kws = {"fmt":"o", "ecolor":"orange", "color":"orange", "alpha":1, "mfc":"orange", "mec":"orange", "mew":1, "zorder":999}

#Masuda et al. 2021 data
specteff = np.array([5996.3610916725,6163.47095871239,6351.57452764171])
specrot= np.array([15.1398954901294,10.9922199999722,5.40119754702563])
spectefferr = np.array([[81.4555633310,58.78236529041,103.289013296011],[103.289013296011,79.77606717985,282.15535339398]])
specroterr = np.array([[5.1308763552480,4.27108719012285,2.39406040514287],[8.1215420545671,7.1261658513948,4.30003441066639]])

for i in range(2):
    arg = hall["Type"] == "MS"
    ax1[i].errorbar(hall["Teff"][arg], hall["P"][arg], 
                   xerr=hall["e_Teff"][arg], 
                   yerr=[hall["e_P"][arg], hall["E_P"][arg]], 
                   label="Hall et al. 2021\nasteroseismic sample", ms=2, lw=0.5, 
                   **ebar_kws)
    ax2[i].errorbar(specteff, specrot, 
                    xerr=spectefferr,
                    yerr=specroterr,
                    label="Masuda et al. 2021", ms=4, lw=1, 
                    **ebar_kws)
    ax1[i].set_xlim(6500,5000)
    #ax1[i].set_ylim(0,60)
    ax1[i].set_xlabel("Effective temperature [K]")
    ax1[i].set_ylabel("Rotation period [d]")
    ax1[i].legend(loc="upper left", prop={"size":9})

    ax2[i].set_xlim(6500,5000)
    ax2[i].set_ylim(0,50)
    ax2[i].set_xlabel("Effective temperature [K]")
    ax2[i].set_ylabel("Rotation period [d]")
    ax2[i].legend(loc="upper left", prop={"size":9})    
    
sns.despine()    
plt.tight_layout()
plt.savefig('../figures/asteroseismic.pdf')
