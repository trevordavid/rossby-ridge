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
# mcq = Table.read(paths.data / 'mcquillan2014/table1.dat',
#                 readme=paths.data / 'mcquillan2014/ReadMe',
#                 format='ascii.cds')
# mcq = mcq.to_pandas()
# mcq = mcq.add_prefix('mcq_')
mcq = pd.read_parquet(paths.data / 'mcquillan2014_table1.parquet')
######################################################################################

######################################################################################
# LAMOST-Kepler 
lam = pd.read_csv(paths.data / 'kepler_lamost.csv')
print('LAMOST unique KIC targets:', len(np.unique(lam["KIC"])))
print('LAMOST unique DR2 targets:', len(np.unique(lam["DR2Name"])))

# Drop duplicate sources, keeping the one with the brighter G magnitude
lam = lam.sort_values(["KIC", "Gmag"], ascending = (True, True))
lam = lam.merge(mcq, how='left', left_on="KIC", right_on="mcq_KIC")
lam = lam.drop_duplicates(subset=['KIC'], keep='first')

lam_mask = (lam["Teff_lam"]>3000)
lam_mask = (lam["Teff_lam"]<8000)
#lam_mask &= (lam["logg_lam"]>3)
lam_mask &= (lam["logg_lam"]>4.1)
lam_mask &= (lam["logg_lam"]<5)
lam_mask &= (abs(lam["feh_lam"])<2)
lam = lam[lam_mask]

print('LAMOST unique KIC targets:', len(np.unique(lam["KIC"])))
print('LAMOST unique DR2 targets:', len(np.unique(lam["DR2Name"])))
print('Median LAMOST Teff error:', np.median(lam["e_Teff_lam"]))
######################################################################################

######################################################################################
#bk = pd.read_csv(paths.data / "_kim_2010/-kim-2010.csv")

def convective_turnover_timescale(teff,
                                  ref='gunn1998'):
    
    #Returns convective turnover timescale in days
    if ref == 'gunn1998':
        #Gunn et al. 1998 relation, from Cranmer & Saar 2011
        return 314.24*np.exp( -(teff/1952.5) - (teff/6250.)**18. ) + 0.002
    
    # elif ref == '2010':
    #     # & Kim 2010 relation for local tau_c
    #     teff_pts = 10.**bk['logT']
    #     tc_pts   = bk['Local_tau_c']
    #     return np.interp(teff, teff_pts, tc_pts)

def constant_rossby(teff, ro):
    #Return locus of rotation periods corresponding to constant Rossby number
    return ro * convective_turnover_timescale(teff)
######################################################################################


# fig, ax = plt.subplots()
# sns.kdeplot(
#     x=lam["Teff_lam"], 
#     y=lam["Prot"], 
#     fill=True, 
#     bw_adjust=0.25,
#     levels=4,
#     #levels=[0.25,0.5,0.75,1],
#     ax=ax
# )


# ax.scatter(lam["Teff_lam"], lam["mcq_Prot"], s=0.1, c='orange', alpha=1, rasterized=True, label='LAMOST–McQuillan')
# ax.set_xlim(5000,7000)
# ax.set_ylim(0,30)



# xcen = 6150
# ycen = constant_rossby(xcen, 1.3)
# xeps = 50
# yeps = 1

# ax.plot([xcen+xeps,xcen+xeps,xcen-xeps,xcen-xeps,xcen+xeps], 
#         [ycen+yeps,ycen-yeps,ycen-yeps,ycen+yeps,ycen+yeps], 'k--')

# _x = np.linspace(5000,6250,100)
# #ax.plot(_x, m*_x + c, 'k--')
# ax.plot(_x, constant_rossby(_x, 1.3), 'k--')
# ax.plot(_x, constant_rossby(_x, 0.5), 'k--')


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
#plt.show()





