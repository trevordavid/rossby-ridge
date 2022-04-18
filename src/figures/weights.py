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

mcq = mcq.merge(gk, how="left", left_on="mcq_KIC", right_on="kepid")
#np.shape(mcq)
mcq = mcq.sort_values(by=["kepid", "kepler_gaia_ang_dist"])
mcq = mcq.drop_duplicates(subset=['kepid'], keep='first')

def lamost_xmatch(df):
    
    df_pos = df[['ra', 'dec']].copy()
    #df_pos.head()
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

sns.set(font_scale=1.3, context="paper", style="ticks")

fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(8,1.5))

for i,j in enumerate(np.arange(0.1,0.31,0.1)):
    
    arg = xm['mcq_w']>j
    axes[i].scatter(xm['Teff'][arg], xm['mcq_Prot'][arg], s=0.05, alpha=0.2, rasterized=True)
    axes[i].set_xlim(7000,4000)
    axes[i].set_ylim(0,40)   
    axes[i].set_title('w > '+"{:.1f}".format(j))
    
    if i>0:
        axes[i].set_xticks([])
        axes[i].set_yticks([])

axes[0].set_xlabel("Effective temperature [K]")
axes[0].set_ylabel("Rotation period [d]")
    
plt.subplots_adjust(wspace=0)
plt.savefig('../figures/weights.pdf')