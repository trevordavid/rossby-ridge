#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as path_effects

import numpy as np

mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300

import seaborn as sns
sns.set_context("paper")

from astropy.table import Table
from astropy import units as u
from astroquery.xmatch import XMatch

cks = pd.read_parquet('../data/cks_merged.parquet')
# The dataframe has a row entry for each KOI, meaning individual star are represented N times
# where N is the number of KOIs detected around that star so we drop duplicates.
print(np.shape(cks))
#cks = cks.drop_duplicates(subset=['kepid'], keep='first')
#print(np.shape(cks))

print(len(cks[np.isfinite(cks['gaia_ra'])]))
print(len(cks[np.isfinite(cks['ra'])]))

cks = cks.dropna(subset=['gaia_ra'])
print(np.shape(cks))

cks_pos = cks[['gaia_ra', 'gaia_dec']].copy()
cks_pos.to_csv('../data/cks-gaia-coords.csv', index=False)




table = XMatch.query(cat1=open('../data/cks-gaia-coords.csv'),
                     cat2='vizier:J/ApJS/245/34/catalog',
                     max_distance=1 * u.arcsec,
                     colRA1='gaia_ra',
                     colDec1='gaia_dec',
                     colRA2='RAJ2000',
                     colDec2='DEJ2000')


table = table.to_pandas()
unq = np.unique(table['gaia_ra'], return_index=True, return_counts=True)

table['xmatch_count'] = np.zeros(len(table))
#table['xmatch_count'].iloc[unq[1]] = unq[2]
for i in range(len(table)):
    arg = unq[0] == table['gaia_ra'].iloc[i]
    table['xmatch_count'].iloc[i] = unq[2][arg]
    
    
table = table.sort_values(by=["gaia_ra"])
table = table.merge(cks, how='right', left_on='gaia_ra', right_on='gaia_ra')
table = table.sort_values(["gaia_ra","angDist"], ascending = (True, True))
table = table.drop_duplicates(subset="gaia_ra", keep="first")



lam = pd.read_csv('../data/kepler_lamost.csv')
#Drop duplicates
lam = lam.drop_duplicates(subset=['KIC'], keep='first')

#McQuillan et al. 2014
# mcq = Table.read('../data/mcquillan2014/table1.dat',
#                 readme='../data/mcquillan2014/ReadMe',
#                 format='ascii.cds')
# mcq = mcq.to_pandas()
# mcq = mcq.add_prefix('mcq_')
mcq = pd.read_parquet('../data/mcquillan2014_table1.parquet')

lam = lam.merge(mcq, how="left", left_on="KIC", right_on="mcq_KIC")


hall = Table.read("https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/table1.dat",
                  readme="https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/ReadMe",
                  format="ascii.cds")

hall = hall.to_pandas()
hall = hall.add_prefix("hall_")

#Gaia-Kepler cross-match from Megan Bedell
gk = pd.read_parquet('../data/kepler_dr2_1arcsec.parquet')
gk = gk.add_prefix('gaia_')

hall = hall.merge(gk,
                 how="left",
                 left_on="hall_KIC",
                 right_on="gaia_kepid")


hall_pos = hall[['gaia_ra', 'gaia_dec']].copy()
hall_pos.to_csv('../data/hall2021-gaia-coords.csv', index=False)


hall_lamost_xmatch = XMatch.query(cat1=open('../data/hall2021-gaia-coords.csv'),
                     cat2='vizier:J/ApJS/245/34/catalog',
                     max_distance=1 * u.arcsec,
                     colRA1='gaia_ra',
                     colDec1='gaia_dec',
                     colRA2='RAJ2000',
                     colDec2='DEJ2000')


hall_lamost_xmatch = hall_lamost_xmatch.to_pandas()
hall_lamost_xmatch = hall_lamost_xmatch.merge(hall, how="left", left_on="gaia_ra", right_on="gaia_ra")


unq = np.unique(hall_lamost_xmatch['gaia_ra'], return_index=True, return_counts=True)

hall_lamost_xmatch['xmatch_count'] = np.zeros(len(hall_lamost_xmatch))

for i in range(len(hall_lamost_xmatch)):
    arg = unq[0] == hall_lamost_xmatch['gaia_ra'].iloc[i]
    hall_lamost_xmatch['xmatch_count'].iloc[i] = unq[2][arg]
    
wavg_teff = np.zeros(len(hall_lamost_xmatch))
avg_teff = np.zeros(len(hall_lamost_xmatch))
med_teff = np.zeros(len(hall_lamost_xmatch))

for i in range(len(hall_lamost_xmatch)):
    gid = hall_lamost_xmatch['gaia_ra'].iloc[i]
    arg = (hall_lamost_xmatch['gaia_ra'] == gid) #& (np.isfinite(table['Teff']))
    _teff = np.array(hall_lamost_xmatch['Teff'][arg])
    _teff_err = np.array(hall_lamost_xmatch['e_Teff'][arg])
    wavg_teff[i] = np.average(_teff, weights=1/_teff_err**2)
    avg_teff[i] = np.mean(_teff)
    med_teff[i] = np.median(_teff)
        
hall_lamost_xmatch['avg_teff'] = avg_teff
hall_lamost_xmatch['wavg_teff'] = wavg_teff
hall_lamost_xmatch['med_teff'] = med_teff        



def one_to_one(ax,
               x1, x2,
               xlabel, ylabel,
               xmin, xmax, sc_kws=None):
    
    ax.plot([xmin,xmax], [xmin,xmax], 'k--', lw=0.5)
    ax.scatter(x1, x2, **sc_kws); 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    
    return ax


def residual(ax,
             x1, x2,
             xlabel, ylabel,
             xmin, xmax, ymin, ymax, sc_kws=None):
    
    
    resid = x1-x2
    
    rms = np.sqrt(np.nanmedian(resid ** 2))

    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.scatter(x1, resid, **sc_kws)

    text1 = ax.text(0.05,0.17,'rms = '+str(int(rms))+' K', transform=ax.transAxes)
    text2 = ax.text(0.05,0.05,'median = '+str(int(np.nanmedian(resid)))+' K', transform=ax.transAxes)
    
    for text in [text1,text2]:
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
       
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    return ax



sns.set(context="paper",
        style="ticks",
        palette="Blues_r",
        font_scale=1.6)


tstr = " Teff [K]"

fig = plt.figure(constrained_layout=False, figsize=(20,5))
widths = [1,1,1,1,1]
heights = [2,1]
spec = fig.add_gridspec(ncols=5, nrows=2, 
                        width_ratios=widths,
                        height_ratios=heights)

xmin,xmax = 4500,6750
ymin,ymax = -700,700

sc_kws = {"alpha": 1, "s":8, "marker":'o', "lw":0, "rasterized":True, "color":"C1"}

ax1 = fig.add_subplot(spec[0,0])
ax2 = fig.add_subplot(spec[1,0])

ax3 = fig.add_subplot(spec[0,1])
ax4 = fig.add_subplot(spec[1,1])

ax5 = fig.add_subplot(spec[0,2])
ax6 = fig.add_subplot(spec[1,2])

ax7 = fig.add_subplot(spec[0,3])
ax8 = fig.add_subplot(spec[1,3])

ax9 = fig.add_subplot(spec[0,4])
ax10 = fig.add_subplot(spec[1,4])

one_to_one(ax1, table["Teff"], table["cks_Teff"], 'LAMOST'+tstr, 'CKS'+tstr, xmin=xmin, xmax=xmax, sc_kws=sc_kws)
residual(ax2, table["Teff"], table["cks_Teff"], 'LAMOST'+tstr, 'LAMOST-\nCKS [K]', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, sc_kws=sc_kws)    

one_to_one(ax3, table["Teff"], table["bf18_Teff"], 'LAMOST'+tstr, 'SPOCS'+tstr, xmin=xmin, xmax=xmax, sc_kws=sc_kws)
residual(ax4, table["Teff"], table["bf18_Teff"], 'LAMOST'+tstr, 'LAMOST-\nSPOCS [K]', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, sc_kws=sc_kws)    

one_to_one(ax5, table["Teff"], table["m19_Teff"], 'LAMOST'+tstr, 'M19'+tstr, xmin=xmin, xmax=xmax, sc_kws=sc_kws)
residual(ax6, table["Teff"], table["m19_Teff"], 'LAMOST'+tstr, 'LAMOST-\nM19 [K]', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, sc_kws=sc_kws)

one_to_one(ax7, lam["Teff_lam"], lam["mcq_Teff"], 'LAMOST'+tstr, 'McQuillan'+tstr, xmin=xmin, xmax=xmax, sc_kws=sc_kws)
residual(ax8, lam["Teff_lam"], lam["mcq_Teff"], 'LAMOST'+tstr, 'LAMOST-\nMcQuillan [K]', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, sc_kws=sc_kws)

one_to_one(ax9, hall_lamost_xmatch["med_teff"], hall_lamost_xmatch["hall_Teff"], 'LAMOST', 'Hall et al. 2021'+tstr, xmin=5000, xmax=xmax, sc_kws=sc_kws)
residual(ax10, hall_lamost_xmatch["med_teff"], hall_lamost_xmatch["hall_Teff"], 'LAMOST', 'LAMOST-\nHall [K]', xmin=5000, xmax=xmax, ymin=ymin, ymax=ymax, sc_kws=sc_kws)


for ax in [ax1,ax3,ax5,ax7,ax9]:
    ax.set_xticklabels([])
    
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig('../figures/teffscales.pdf')