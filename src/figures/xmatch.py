import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300

import seaborn as sns
sns.set_context("paper")

from astropy.table import Table
import astropy.units as u
import astropy.constants as c
from astroquery.xmatch import XMatch
import pandas as pd


#McQuillan et al. 2014
mcq = pd.read_parquet('../data/mcquillan2014_table1.parquet')

#Gaia-Kepler cross-match from Megan Bedell
gk = pd.read_parquet('../data/kepler_dr2_1arcsec.parquet')
#gk = gk.to_pandas()
#gk.head()

mcq = mcq.merge(gk, how="left", left_on="mcq_KIC", right_on="kepid")
#np.shape(mcq)
mcq = mcq.sort_values(by=["kepid", "kepler_gaia_ang_dist"])
mcq = mcq.drop_duplicates(subset=['kepid'], keep='first')

def lamost_xmatch(df):
    
    df_pos = df[['ra', 'dec']].copy()
    df_pos.head()
    df_pos.to_csv('../data/coords.csv', index=False)
    
    table = XMatch.query(cat1=open('../data/coords.csv'),
                         cat2='vizier:J/ApJS/245/34/catalog',
                         max_distance=1 * u.arcsec,
                         colRA1='ra',
                         colDec1='dec',
                         colRA2='RAJ2000',
                         colDec2='DEJ2000')

    #type(table)
    #print(table)
    table = table.to_pandas()
    
    unq = np.unique(table['ra'], return_index=True, return_counts=True)

    table['xmatch_count'] = np.zeros(len(table))

    for i in range(len(table)):
        arg = unq[0] == table['ra'].iloc[i]
        table['xmatch_count'].iloc[i] = unq[2][arg]

    #table = table.sort_values(by=["gaia_ra"])
    table = table.merge(df, how='right', left_on='ra', right_on='ra')
    table = table.sort_values(["ra","angDist"], ascending = (True, True))
    table = table.drop_duplicates(subset="ra", keep="first")
    
    return table
    
    
xm = lamost_xmatch(mcq)

print(len(xm), 'unique stars in LAMOST-McQuillan cross-match')
arg = (np.isfinite(xm["Teff"]) & (np.isfinite(xm["mcq_Prot"])))
print(len(xm[arg]), 'unique stars in LAMOST-McQuillan cross-match with Teff and Prot')


sns.set(font_scale=1.3, context="paper", style="ticks")

sun = {"teff": 5772,
       "prot": 25.4,
       "e_prot": 25.4-24.5,
       "E_prot": 36-25.4
      }

sun["logg"] = np.log10(c.GM_sun.cgs.value/c.R_sun.cgs.value**2)

x = xm["Teff"]
y = xm["mcq_Prot"]

arg = (np.isfinite(x)) & (np.isfinite(y))
ms  = arg & (xm["logg_x"]>4.2)
sg  = arg & (xm["logg_x"]<4.2)

titles = ["LAMOST-McQuillan\nall stars",
          "LAMOST-McQuillan\nlog(g) < 4.2 (subgiants)",
          "LAMOST-McQuillan\nlog(g) > 4.2 (dwarfs)"]

h_kws = {"bins":150, "cmap": "Blues", "cmin": 1}

fig,axes = plt.subplots(nrows=1,
                        ncols=3,
                        figsize=(12,3))
cb1 = axes[0].hist2d(x[arg], y[arg], **h_kws)
cb2 = axes[1].hist2d(x[sg], y[sg], **h_kws)
cb3 = axes[2].hist2d(x[ms], y[ms], **h_kws)

for i,cb in enumerate([cb1,cb2,cb3]):
    fig.colorbar(cb[3], ax=axes[i], label=r"N$_\mathregular{stars}$")

axes[0].text(0.7,0.9,str(len(x[arg]))+' stars', transform=axes[0].transAxes, size=9)
axes[1].text(0.7,0.9,str(len(x[sg]))+' stars', transform=axes[1].transAxes, size=9)
axes[2].text(0.7,0.9,str(len(x[ms]))+' stars', transform=axes[2].transAxes, size=9)

for i,ax in enumerate(axes):
    ax.set_xlim(7000,4000)
    ax.set_ylim(0,40)
    ax.errorbar(sun["teff"], sun["prot"], yerr=np.vstack([sun["e_prot"], sun["E_prot"]]), fmt="o", 
                   color="C1", mec="white", ms=6)
    ax.set_title(titles[i])
    ax.set_xlabel('Effective temperature [K]')
    ax.set_ylabel('Rotation period [d]')

sns.despine()
plt.tight_layout()
plt.savefig('../figures/lamost-mcquillan.pdf')
#plt.show()



#Santos et al. 2021
san = pd.read_csv('../data/santos2021/S21_rotators.csv')
print(len(san), 'stars before removing duplicates')
san = san.add_prefix('san_')
san = san.merge(gk, how="left", left_on="san_KIC", right_on="kepid")
san = san.sort_values(by=["kepid", "kepler_gaia_ang_dist"])
san = san.drop_duplicates(subset=['kepid'], keep='first')
print(len(san), 'stars after removing duplicates')
san.head()
    
xm = lamost_xmatch(san)

print(len(xm), 'unique stars in LAMOST-Santos cross-match')
arg = (np.isfinite(xm["Teff"])&(np.isfinite(xm["san_Prot"])))
print(len(xm[arg]), 'unique stars in LAMOST-Santos cross-match with Teff and Prot')

x = xm["Teff"]
y = xm["san_Prot"]

arg = (np.isfinite(x)) & (np.isfinite(y))
ms  = arg & (xm["logg_x"]>4.2)
sg  = arg & (xm["logg_x"]<4.2)

titles = ["LAMOST-Santos\nall stars",
          "LAMOST-Santos\nlog(g) < 4.2 (subgiants)",
          "LAMOST-Santos\nlog(g) > 4.2 (dwarfs)"]

h_kws = {"bins":200, "cmap": "Blues", "cmin": 1, "density":False}

fig,axes = plt.subplots(nrows=1,
                        ncols=3,
                        figsize=(12,3))
cb1 = axes[0].hist2d(x[arg], y[arg], **h_kws)
cb2 = axes[1].hist2d(x[sg], y[sg], **h_kws)
cb3 = axes[2].hist2d(x[ms], y[ms], **h_kws)


axes[0].text(0.7,0.9,str(len(x[arg]))+' stars', transform=axes[0].transAxes, size=9)
axes[1].text(0.7,0.9,str(len(x[sg]))+' stars', transform=axes[1].transAxes, size=9)
axes[2].text(0.7,0.9,str(len(x[ms]))+' stars', transform=axes[2].transAxes, size=9)

for i,cb in enumerate([cb1,cb2,cb3]):
    fig.colorbar(cb[3], ax=axes[i], label=r"N$_\mathregular{stars}$")
    

for i,ax in enumerate(axes):
    ax.set_xlim(7000,4000)
    ax.set_ylim(0,40)
    ax.errorbar(sun["teff"], sun["prot"], yerr=np.vstack([sun["e_prot"], sun["E_prot"]]), fmt="o", 
                   color="C1", mec="white", ms=6)
    ax.set_title(titles[i])
    ax.set_xlabel('Effective temperature [K]')
    ax.set_ylabel('Rotation period [d]')

sns.despine()
plt.tight_layout()
plt.savefig('../figures/lamost-santos.pdf')
#plt.show()