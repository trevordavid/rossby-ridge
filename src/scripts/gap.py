#!/usr/bin/env python
# coding: utf-8

import paths
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
       "prot":25.4,
       "e_prot":0,
       "E_prot":27-25.4
      }

sun["logg"] = np.log10(c.GM_sun.cgs.value/c.R_sun.cgs.value**2)

######################################################################################
#McQuillan et al. 2014
# mcq = Table.read(paths.data / 'mcquillan2014/table1.dat',
#                 readme=paths.data / 'mcquillan2014/ReadMe',
#                 format='ascii.cds')
# mcq = mcq.to_pandas()
# mcq = mcq.add_prefix('mcq_')
mcq = pd.read_parquet(paths.data / 'mcquillan2014_table1.parquet')
######################################################################################

######################################################################################
#Gaia-Kepler cross-match from Megan Bedell
gk = pd.read_parquet(paths.data / 'kepler_dr2_1arcsec.parquet')
######################################################################################

######################################################################################
# LAMOST-Kepler 
lam = pd.read_csv(paths.data / 'kepler_lamost.csv')
print('LAMOST unique KIC targets:', len(np.unique(lam["KIC"])))
print('LAMOST unique DR2 targets:', len(np.unique(lam["DR2Name"])))

# Drop duplicate sources, keeping the one with the brighter G magnitude
lam = lam.sort_values(["KIC", "Gmag"], ascending = (True, True))

#Merge
lam = lam.merge(mcq, how="left", left_on="KIC", right_on="mcq_KIC")
lam = lam.merge(gk, how="left", left_on="KIC", right_on="kepid")

lam = lam.drop_duplicates(subset=['KIC'], keep='first')

lam_mask = (lam["Teff_lam"]>3000)
lam_mask = (lam["Teff_lam"]<8000)
lam_mask &= (lam["logg_lam"]>3)
lam_mask &= (lam["logg_lam"]<5)
lam_mask &= (abs(lam["feh_lam"])<2)
#lam_mask &= (lam["mcq_Rper"]>2)
#lam_mask &= (lam["phot_g_mean_mag"]<15)
#lam_mask &= (lam["r_est"]>0.) & (lam["r_est"]<500.)
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

lam["Ro"] = lam["Prot"]/convective_turnover_timescale(lam["Teff_lam"])    
mcq["Ro"] = mcq["mcq_Prot"]/convective_turnover_timescale(mcq["mcq_Teff"])    


from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
bottom = cm.get_cmap('Oranges', 128)
top = cm.get_cmap('Blues_r', 128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
OrBu = ListedColormap(newcolors, name='OrangeBlue')


sns.set(font_scale=1.6, context="paper", style="ticks")
sc_kws = {"marker":".", 
          "alpha": 0.5, 
          "rasterized":True, 
          "cmap": OrBu,
          "s": 40,
          "vmin":3,
          "vmax":4.5,
          "lw":0}


fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2,
                             figsize=(14,5))

logrper = np.log10(lam["mcq_Rper"])

lam_arg = (abs(np.log10(lam["Ro"]))<1)

sns.kdeplot(lam["Teff_lam"], 
            lam["Prot"],
            bw_adjust=0.6, ax=ax1, lw=0.1, color='k', alpha=0.5)

sns.kdeplot(mcq["mcq_Teff"], 
            mcq["mcq_Prot"], 
            bw_adjust=0.6, ax=ax2, lw=0.1, color='k', alpha=0.5)

cb1 = ax1.scatter(lam["Teff_lam"], 
            lam["Prot"], 
            c=logrper, **sc_kws)

cb2 = ax2.scatter(mcq["mcq_Teff"], 
            mcq["mcq_Prot"], 
            c=np.log10(mcq["mcq_Rper"]), 
            **sc_kws)



for ax in [ax1,ax2]:
    _teff = np.linspace(3000,7000,1000)
    ax.plot(_teff, constant_rossby(_teff, 0.5), '--', color='k', lw=2, alpha=0.5)
    ax.set_ylim(0,50)  
    ax.set_xlabel("Effective temperature [K]")
    ax.set_ylabel("Rotation period [d]")    

ax1.set_xlim(7000,4000)
ax2.set_xlim(7000,3000)
    
ax1.set_title('LAMOST–McQuillan')
ax2.set_title('McQuillan et al. 2014')

plt.colorbar(cb1, label=r'log(R$_\mathregular{per}$/ppm) [dex]', ax=ax1)    
plt.colorbar(cb2, label=r'log(R$_\mathregular{per}$/ppm) [dex]', ax=ax2)

plt.tight_layout()
sns.despine()
plt.savefig(paths.figures / 'gap.pdf')
#plt.show()