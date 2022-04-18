#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300

import seaborn as sns
import astropy.constants as c
from scipy import interpolate
from labellines import labelLine, labelLines

sun = {"teff": 5772,
       "prot": 25.4,
       "e_prot": 25.4-24.5,
       "E_prot": 27-25.4
      }

sun["logg"] = np.log10(c.GM_sun.cgs.value/c.R_sun.cgs.value**2)


def convective_turnover_timescale(teff):
    #Returns convective turnover timescale in days
    #Gunn et al. 1998 relation, from Cranmer & Saar 2011
    return 314.24*np.exp(-(teff/1952.5) - (teff/6250.)**18.) + 0.002


def constant_rossby(teff, ro):
    #Return locus of rotation periods corresponding to constant Rossby number
    return ro * convective_turnover_timescale(teff)

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
    f = interpolate.interp1d(_teff[_], _prot[_], kind='cubic', fill_value='extrapolate')
    
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


# from scipy.interpolate import interp1d

# std = pd.read_hdf('../data/models/standard_population.h5', key='sample')
# #roc = pd.read_hdf('../models/rocrit_population.h5', key='sample')

# idx = (abs(std['age']-5)<0.01) & (std['evo']==1) & (abs(std['[Fe/H]'])<0.05)
# model_teff = np.array(std['Teff'][idx])
# model_prot = np.array(std['period'][idx])

# order = np.argsort(model_teff)
# model_teff = model_teff[order]
# model_prot = model_prot[order]

# df = pd.DataFrame({"x": model_teff,
#                    "y": model_prot})
# df = df.drop_duplicates()

# x = np.array(df.x)
# y = np.array(df.y)
# f = interp1d(x, y, kind='linear', fill_value='extrapolate')


#McQuillan et al. 2014
mcq = pd.read_parquet('../data/mcquillan2014_table1.parquet')


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
######################################################################################

# ######################################################################################
# hall = Table.read("https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/table1.dat",
#                   readme="https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/ReadMe",
#                   format="ascii.cds")

# hall.info()
# ######################################################################################

sns.set(style='ticks', font_scale=1.5, context='paper')

_teff = np.linspace(5000,6250,1000)

seq = 'ngc6819+ruprecht147'

#Masuda et al. 2021 data
specteff = np.array([5996.3610916725,6163.47095871239,6351.57452764171])
specrot= np.array([15.1398954901294,10.9922199999722,5.40119754702563])
spectefferr = np.array([[81.4555633310,58.78236529041,103.289013296011],[103.289013296011,79.77606717985,282.15535339398]])
specroterr = np.array([[5.1308763552480,4.27108719012285,2.39406040514287],[8.1215420545671,7.1261658513948,4.30003441066639]])

sns.displot(x=lam["Teff_lam"], 
            y=lam["Prot"], binwidth=(50, 1))

plt.errorbar(specteff, specrot, xerr=spectefferr, yerr=specroterr, fmt='o', color='white', zorder=np.inf)

sun_kws = {"marker":"o", "color":"black", "ms":8, "mfc":"None", "mew":1}
plt.plot(sun["teff"], sun["prot"], **sun_kws)
plt.plot(sun["teff"], sun["prot"], 'k.')

plt.plot(_teff, curtis_teff_gyrochrone(_teff, kind=seq)*(5./2.5)**0.65, label='5 Gyr (n=0.65)', color='orange', lw=4, ls=':')
#plt.plot(_teff, f(_teff), label='model', color='orange', lw=4, ls=':')
plt.plot(_teff, curtis_teff_gyrochrone(_teff, kind=seq)*(5./2.5)**0.5, label='5 Gyr (n=0.5)', color='orange', lw=4, ls='--')
plt.plot(_teff, curtis_teff_gyrochrone(_teff, kind=seq), label='2.5 Gyr', color='orange', lw=4)
plt.legend()
plt.xlim(7000,5000)
plt.ylim(0,50)
plt.xlabel("Effective temperature [K]")
plt.ylabel("Rotation period [d]")
plt.savefig('../figures/skumanich.pdf')