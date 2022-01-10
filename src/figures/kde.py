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
       "E_prot": 36-25.4
      }

sun["logg"] = np.log10(c.GM_sun.cgs.value/c.R_sun.cgs.value**2)

#bk = pd.read_csv("../data/_kim_2010/-kim-2010.csv")

def convective_turnover_timescale(teff,
                                  ref='gunn1998'):
    #Returns convective turnover timescale in days
    
    if ref == 'gunn1998':
        #Gunn et al. 1998 relation, from Cranmer & Saar 2011
        return 314.24*np.exp(-(teff/1952.5) - (teff/6250.)**18.) + 0.002
    
    # elif ref == '2010':
    #     # & Kim 2010 relation for local tau_c
    #     teff_pts = 10.**bk['logT']
    #     tc_pts   = bk['Local_tau_c']
    #     return np.interp(teff, teff_pts, tc_pts)


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
hall = Table.read("https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/table1.dat",
                  readme="https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/ReadMe",
                  format="ascii.cds")

hall.info()
######################################################################################

sns.set(style='ticks', font_scale=1.4, context='paper')


# In[4]:


sns.set(style='ticks', font_scale=1.6, context='paper')

fig,(ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, 
                            figsize=(15,6))

sns.kdeplot(
    x=cks["cks_Teff"], 
    y=cks["mcq_Prot"], 
    fill=True, 
    bw_adjust=0.5,
    ax=ax1
)

sns.kdeplot(
    x=lam["Teff_lam"], 
    y=lam["Prot"], 
    fill=True, 
    bw_adjust=0.25,
    ax=ax2
)

sns.kdeplot(
    x=hall["Teff"], 
    y=hall["P"], 
    fill=True, 
    bw_adjust=0.5,
    ax=ax3
)


for ax in [ax1,ax2,ax3]:
    
    ax.set_xlim(6750,4500)
    ax.set_ylim(-1,41)
    ax.set_xlabel("Effective temperature [K]")
    ax.set_ylabel("Rotation period [d]")
    

    gyro_sequences = ['pleiades-ro', 'praesepe', 'ngc6811', 'ngc6819+ruprecht147']
    gyro_ages = ['0.12 Gyr', '0.67 Gyr', '1 Gyr', '2.5 Gyr']
    _teff = np.linspace(4500,6250,1000)

    for i,seq in enumerate(gyro_sequences):
        ax.plot(_teff, curtis_teff_gyrochrone(_teff, kind=seq), label=gyro_ages[i], color='k', lw=3, alpha=0.5)
 
    for i,_ro in enumerate([0.4,1.45,2]):
        ax.plot(_teff, constant_rossby(_teff, _ro), 'orange', lw=3, ls='--', alpha=0.5, label="Ro = "+str(_ro))                
        
    labelLines(ax.get_lines(), 
               outline_color='#eeeeee',
               outline_width=3,
               xvals=(4500, 5600), 
               zorder=2.5, 
               size=9)  
    
    
    ax.plot(sun["teff"], sun["prot"], 'o', color='C1', label='Sun')
    ax.errorbar(sun["teff"], sun["prot"], yerr=np.vstack([sun["e_prot"], sun["E_prot"]]), fmt="o", 
                   color="C1", mec="white", ms=6)
    

ax1.set_title('CKS–McQuillan')
ax2.set_title('LAMOST–McQuillan')
ax3.set_title('Hall et al. 2021')
sns.despine()
plt.tight_layout()
plt.savefig('../figures/kde.pdf')
