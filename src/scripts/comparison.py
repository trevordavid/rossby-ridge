import paths
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import numpy as np

mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300

import seaborn as sns
sns.set_context("paper")

from astropy.table import Table

def ridge_hi(teff):
    m = (2-24)/(6500-5800)
    b = (2 - m*6500) 
    return m*teff + b

def ridge_lo(teff):
    m = (2-24)/(6500-5800)
    b = (-5 - m*6500) 
    return m*teff + b

# Parallelogram
xp = [6500,6500,5800,5800]
yp = [ridge_lo(6500),ridge_hi(6500),ridge_hi(5800),ridge_lo(5800)]

cks = pd.read_parquet(paths.data / 'cks_merged.parquet')
# The dataframe has a row entry for each KOI, meaning individual star are represented N times
# where N is the number of KOIs detected around that star so we drop duplicates.
#print(np.shape(cks))
#cks = cks.drop_duplicates(subset=['kepid'], keep='first')
#print(np.shape(cks))

#Santos et al. 2021
#san = pd.read_csv(paths.data / 'S21_rotators.csv')
san = pd.read_parquet(paths.data / 'santos2021_rotators.parquet')
san = san.add_prefix('san_')
san = san.sort_values(['san_KIC', 'san_Kp'], ascending = (True, True))
san = san.drop_duplicates(subset=['san_KIC'], keep='first')
cks = cks.merge(san, left_on='kepid', right_on='san_KIC', how='left')


print(len(cks[np.isfinite(cks['gaia_ra'])]))
print(len(cks[np.isfinite(cks['ra'])]))

cks = cks.dropna(subset=['gaia_ra'])
print(np.shape(cks))


cks_pos = cks[['gaia_ra', 'gaia_dec']].copy()
cks_pos.to_csv(paths.data / 'cks-gaia-coords.csv', index=False)




from astropy import units as u
from astroquery.xmatch import XMatch
import pandas as pd
table = XMatch.query(cat1=open(paths.data / 'cks-gaia-coords.csv'),
                     cat2='vizier:J/ApJS/245/34/catalog',
                     max_distance=1 * u.arcsec,
                     colRA1='gaia_ra',
                     colDec1='gaia_dec',
                     colRA2='RAJ2000',
                     colDec2='DEJ2000')

type(table)
print(table)

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


sns.set(font_scale=1.1, context="paper", style="ticks", palette="Blues_r")

fig, axes = plt.subplots(nrows=5,
                         ncols=4,
                         figsize=(12,12))

sc_kws = {"s":1, "color":"C1", "alpha":0.5, "rasterized":True}


prot = [table['san_Prot'],
        table['d21_prot'],
        table['a18_period'],
        table['m15_Prot'],
        table['m13_Prot']]

for i,pr in enumerate(prot):
    axes[i][0].scatter(table['Teff'], prot[i], **sc_kws)
    axes[i][1].scatter(table['cks_Teff'], prot[i], **sc_kws)
    axes[i][2].scatter(table['bf18_Teff'], prot[i], **sc_kws)
    axes[i][3].scatter(table['m19_Teff'], prot[i], **sc_kws)
    
for i in range(5):
    for j in range(4):
        #if i<3:
        #    axes[i][j].set_xticklabels('')
        #if j>0:
        #    axes[i][j].set_yticklabels('')
            
        axes[i][j].set_xlim(7000,4100)
        axes[i][j].set_ylim(-2,65)
       
        #Parallelogram
        #axes[i][j].add_patch(patches.Polygon(xy=list(zip(xp,yp)), fill=False, lw=1, color='k'))

        
axes[4][0].set_xlabel('Effective temperature [K]\n(LAMOST; Xiang et al. 2019)')
axes[4][1].set_xlabel('Effective temperature [K]\n(CKS; Fulton & Petigura 2018)')
axes[4][2].set_xlabel('Effective temperature [K]\n(SPOCS; Brewer & Fischer 2018)')
axes[4][3].set_xlabel('Effective temperature [K]\n(Martinez et al. 2019)')

axes[0][0].set_ylabel('Rotation period [d]\n(Santos et al. 2021)')
axes[1][0].set_ylabel('Rotation period [d]\n(David et al. 2021)')
axes[2][0].set_ylabel('Rotation period [d]\n(Angus et al. 2018)')
axes[3][0].set_ylabel('Rotation period [d]\n(Mazeh et al. 2015)')
axes[4][0].set_ylabel('Rotation period [d]\n(McQuillan et al. 2013)')

sns.despine()
plt.subplots_adjust(hspace=0.25,wspace=0.25)
plt.savefig(paths.figures / 'comparison.pdf')