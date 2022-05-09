import paths
import scipy.stats as st
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
mcq = pd.read_parquet(paths.data / 'mcquillan2014_table1.parquet')

#Gaia-Kepler cross-match from Megan Bedell
gk = pd.read_parquet(paths.data / 'kepler_dr2_1arcsec.parquet')

mcq = mcq.merge(gk, how="left", left_on="mcq_KIC", right_on="kepid")
#np.shape(mcq)
mcq = mcq.sort_values(by=["kepid", "kepler_gaia_ang_dist"])
mcq = mcq.drop_duplicates(subset=['kepid'], keep='first')

def lamost_xmatch(df):
    
    df_pos = df[['ra', 'dec']].copy()
    #df_pos.head()
    df_pos.to_csv(paths.data / 'coords.csv', index=False)
    
    table = XMatch.query(cat1=open(paths.data / 'coords.csv'),
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

#print(len(xm), 'unique stars in LAMOST-McQuillan cross-match')
#arg = (np.isfinite(xm["Teff"]) & (np.isfinite(xm["mcq_Prot"])))
#print(len(xm[arg]), 'unique stars in LAMOST-McQuillan cross-match with Teff and Prot')

sns.set(font_scale=1.3, context="paper", style="ticks")

x = xm["Teff"]
y = xm["mcq_Prot"]

logg_thresh = 4.1

arg = (np.isfinite(x)) & (np.isfinite(y))
ms  = arg & (xm["logg_x"]>logg_thresh)
sg  = arg & (xm["logg_x"]<logg_thresh)


def convective_turnover_timescale(teff):
    #Returns convective turnover timescale in days
    #Gunn et al. 1998 relation, from Cranmer & Saar 2011
    return 314.24*np.exp(-(teff/1952.5) - (teff/6250.)**18.) + 0.002
    

def constant_rossby(teff, ro):
    #Return locus of rotation periods corresponding to constant Rossby number
    return ro * convective_turnover_timescale(teff)

def gaussian_kde(x, y):
    
    arg = (np.isfinite(x)) & (np.isfinite(y))
    x = x[arg]
    y = y[arg]

    xmin, xmax = 0.9*x.min(),1.1*x.max()
    ymin, ymax = 0.9*y.min(),1.1*y.max()

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    return xx, yy, f


_teff = np.linspace(4500,7000,1000)
_roc  = constant_rossby(_teff, 1.25)

h_kws = {"bins":200, "cmap": "Blues", "cmin": 1}
line_kws = {"color":"orange", "lw":2}

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(3.5,3.5))

sns.kdeplot(
    x=x[ms], 
    y=y[ms], 
    fill=True, 
    bw_adjust=0.2,
    ax=ax,
    cmap='Blues'
)

ax.plot(_teff, _roc, '-', **line_kws, label='Ro=1.25')
ax.plot(_teff, _roc/2., '--', **line_kws, label=r'$\mathregular{\frac{1}{2} \times}$ (Ro=1.25)')
ax.plot(_teff, _roc/3., ':', **line_kws, label=r'$\mathregular{\frac{1}{3} \times}$ (Ro=1.25)')

ax.set_xlim(7000,5000)
ax.set_ylim(0,30)

ax.set_xlabel('Effective temperature [K]')
ax.set_ylabel('Rotation period [d]')
ax.legend(prop={"size":10})

plt.tight_layout()
plt.savefig(paths.figures / 'harmonic.pdf')