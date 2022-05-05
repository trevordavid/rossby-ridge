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

from scipy import interpolate

import astropy.constants as c

sun = {"teff": 5772,
       "prot": 25.4,
       "e_prot": 25.4-24.5,
       "E_prot": 27-25.4
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

############################################
cks_teff = cks["p20_cks_steff"]
cks_e_teff = cks["p20_cks_steff_err1"]
cks_prot = cks["d21_prot"]

def ridge_hi(teff):
    m = (2-24)/(6500-5800)
    b = (2 - m*6500) 
    return m*teff + b

def ridge_lo(teff):
    m = (2-24)/(6500-5800)
    b = (-5 - m*6500) 
    return m*teff + b

mask = (cks['p20_cks_slogg']>4.1) #main sequence
ridge = (cks['p20_cks_steff']>5850)
ridge &= (cks['p20_cks_steff']<6500)
ridge &= (cks['d21_prot']<ridge_hi(cks['p20_cks_steff']))
ridge &= (cks['d21_prot']>ridge_lo(cks['p20_cks_steff']))
ridge &= mask
############################################

######################################################################################
#bk = pd.read_csv("../data/_kim_2010/-kim-2010.csv")

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


#Models
std = pd.read_hdf('../data/standard_population.h5', key='sample')
std['ro'] = std['period']/(std['taucz']/86400)
std = std[std['evo']==1] # limit to main-sequence

roc = pd.read_hdf('../data/rocrit_population.h5', key='sample')
roc['ro'] = roc['period']/(roc['taucz']/86400)
roc = roc[roc['evo']==1] # limit to main-sequence




fig, ax = plt.subplots()
sns.kdeplot(
    x=lam["Teff_lam"], 
    y=lam["Prot"], 
    fill=True, 
    bw_adjust=0.25,
    levels=4,
    #levels=[0.25,0.5,0.75,1],
    ax=ax
)

ax.scatter(lam["Teff_lam"], lam["mcq_Prot"], s=0.1, c='orange', alpha=1, rasterized=True, label='LAMOST–McQuillan')
ax.set_xlim(5000,7000)
ax.set_ylim(0,30)

_x = np.linspace(5000,6250,100)
ax.plot(_x, constant_rossby(_x, 1.3), 'k--')
ax.plot(_x, constant_rossby(_x, 0.5), 'k--')


dist = abs(lam["Prot"] - constant_rossby(lam["Teff_lam"], 1.3))
frac_dist = abs(lam["Prot"] - constant_rossby(lam["Teff_lam"], 1.3))/constant_rossby(lam["Teff_lam"], 1.3)
lam_ridge = (frac_dist<0.05) & (lam["Teff_lam"]>5500) & (lam["Teff_lam"]<6500) & (lam["logg_lam"]>4.1) & (lam["logg_lam"]<4.75)

ax.plot(lam["Teff_lam"][lam_ridge], lam["Prot"][lam_ridge], 'o', mfc="None", color='white', alpha=0.2);


sns.set(font_scale=1.1, context="paper", style="ticks")

mpl.rcParams["legend.markerscale"] = 2

all_f = mask & (cks['p20_cks_steff']>5800) & (cks['p20_cks_slogg']<4.5) & (cks['p20_cks_steff']<6500)

fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(8,6))

ax = axes[0]

ax[0].plot(cks['p20_cks_steff'], cks['p20_cks_slogg'], '.', color='lightgrey', label='California–Kepler Survey', rasterized=True, ms=2)
ax[0].plot(cks['p20_cks_steff'][ridge], cks['p20_cks_slogg'][ridge], '.', color='k', label='Long-period pile-up', rasterized=True, ms=2)
ax[0].errorbar(5000, 4.8, xerr=np.nanmedian(cks['p20_cks_steff_err1']),
                          yerr=np.nanmedian(cks['p20_cks_slogg_err1']), fmt='.', color='k', zorder=1000)


ax[1].plot(lam['Teff_lam'], lam['logg_lam'], 
           '.', color='lightgrey', label='LAMOST–McQuillan', rasterized=True, ms=2)
ax[1].plot(lam['Teff_lam'][lam_ridge], lam['logg_lam'][lam_ridge], 
           '.', color='k', label='Long-period pile-up', rasterized=True, ms=2)
ax[1].errorbar(5000, 4.8, xerr=np.nanmedian(lam['e_Teff_lam']),
                          yerr=np.nanmedian(lam['e_logg_lam']), fmt='.', color='k', zorder=1000)


ax[0].plot([5850,6500,6500,5850,5850],[4.1,4.1,4.75,4.75,4.1],color='k',lw=0.5)
ax[1].plot([5500,6500,6500,5500,5500],[4.1,4.1,4.75,4.75,4.1],color='k',lw=0.5)

for i in range(2):
    
    ax[i].set_xlim(6750,4500)
    ax[i].set_ylim(5,3)
    ax[i].set_ylabel('log(g) [dex]')
    ax[i].set_xlabel('Effective temperature [K]')
    
    ax[i].text(4900, 4.9, 'typical\nerror', size=10)
    ax[i].plot(sun["teff"], sun["logg"], 'o', color='orange', label='Sun')
    lgnd = ax[i].legend(prop={'size':10}, loc='upper left')
      
    
for i,let in enumerate("ab"):
    ax[i].text(1.05,1.05,let,transform=ax[i].transAxes,weight='bold')    
    
ax = axes[1]    


cks['cks_e_age'] = cks['cks_age'] - (10.**(cks['cks_logAiso']-cks['cks_e_logAiso'])/1.0e9)
cks['cks_E_age'] = (10.**(cks['cks_logAiso']+cks['cks_E_logAiso'])/1.0e9) - cks['cks_age']

# ax[0].errorbar(cks['p20_cks_steff'], cks['cks_age'], 
#              xerr=[cks['cks_e_Teff'], cks['cks_E_Teff']],
#              yerr=[cks['cks_e_age'], cks['cks_E_age']], fmt='o', color='lightgrey',mec='lightgrey', linewidth=0, ecolor='lightgrey', zorder=1, alpha=0.5)

# ax[0].errorbar(cks['p20_cks_steff'][ridge], cks['cks_age'][ridge], 
#              xerr=[cks['cks_e_Teff'][ridge], cks['cks_E_Teff'][ridge]],
#              yerr=[cks['cks_e_age'][ridge], cks['cks_E_age'][ridge]], fmt='o', mec='white', linewidth=0, color='k', ecolor='k', zorder=2)

cks_age_err = np.max([np.nanmedian(cks['cks_e_age']), np.nanmedian(cks['cks_E_age'])])
cks_teff_err = np.nanmedian(cks['p20_cks_steff_err1'])
spocs_age_err = np.max([np.nanmedian(cks['bf18_e_Age']), np.nanmedian(cks['bf18_E_Age'])])
spocs_teff_err = np.nanmedian(cks['bf18_e_Teff'])


ax[0].plot(cks['p20_cks_steff'], cks['cks_age'], 
            '.', color='lightgrey', zorder=1)

#Plot models
for i in range(2):
    ax[i].plot(roc['Teff'][roc['ro']>2], roc['age'][roc['ro']>2], 
                ',', ms=0.2, color='orange', alpha=0.5, zorder=1, rasterized=True)
    ax[i].plot(roc['Teff'][roc['ro']<2], roc['age'][roc['ro']<2], 
                ',', ms=0.2, color='C0', alpha=0.5, zorder=1, rasterized=True)            

ax[0].plot(cks['p20_cks_steff'][ridge], cks['cks_age'][ridge], 
            '.', color='k', zorder=2)

ax[0].errorbar(6500, 10, xerr=cks_teff_err,
                          yerr=cks_age_err, fmt='.', color='k', zorder=1000)

ax[0].set_ylabel('CKS Age [Gyr]')
ax[0].set_xlabel('CKS Effective temperature [K]')


# ax[1].errorbar(cks['p20_cks_steff'], cks['bf18_Age'], 
#              xerr=[cks['cks_e_Teff'], cks['cks_E_Teff']],
#              yerr=[cks['bf18_e_Age'], cks['bf18_E_Age']], fmt='o', color='lightgrey', mec='lightgrey', linewidth=0, ecolor='lightgrey', alpha=0.5,  zorder=1)

# ax[1].errorbar(cks['p20_cks_steff'][ridge], cks['bf18_Age'][ridge], 
#              xerr=[cks['cks_e_Teff'][ridge], cks['cks_E_Teff'][ridge]],
#              yerr=[cks['bf18_e_Age'][ridge], cks['bf18_E_Age'][ridge]], fmt='o', mec='white', linewidth=0, color='k', ecolor='k', zorder=2)

ax[1].plot(cks['bf18_Teff'], cks['bf18_Age'], 
'.', color='lightgrey', zorder=1)

ax[1].plot(cks['bf18_Teff'][ridge], cks['bf18_Age'][ridge],
'.', color='k', zorder=2)

ax[1].errorbar(6500, 10, xerr=spocs_teff_err,
                          yerr=spocs_age_err, fmt='.', color='k', zorder=1000)

ax[1].set_ylabel('SPOCS Age [Gyr]')
ax[1].set_xlabel('SPOCS Effective temperature [K]')

for i in range(2):
    ax[i].set_xlim(6600,5600)
    ax[i].set_ylim(0,12)
    ax[i].scatter(sun["teff"], 4.567, color='orange', zorder=3)
    ax[i].text(6450, 11, 'typical error', size=10)
    
for i,let in enumerate("cd"):
    ax[i].text(1.05,1.05,let,transform=ax[i].transAxes,weight='bold')
    
plt.tight_layout()
#sns.despine()
plt.savefig('../figures/ages.pdf')

print('5th and 95th percentile range of CKS ages (Gyr)   :', np.nanpercentile(cks['cks_age'][ridge], [5,95]))
print('5th and 95th percentile range of SPOCS ages (Gyr) :', np.nanpercentile(cks['bf18_Age'][ridge], [5,95]))
