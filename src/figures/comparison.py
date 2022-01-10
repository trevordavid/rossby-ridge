#!/usr/bin/env python
# coding: utf-8

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
cks.head()


cks_pos = cks[['gaia_ra', 'gaia_dec']].copy()
cks_pos.head()
cks_pos.to_csv('../data/cks-gaia-coords.csv', index=False)


from astropy import units as u
from astroquery.xmatch import XMatch
import pandas as pd
table = XMatch.query(cat1=open('../data/cks-gaia-coords.csv'),
                     cat2='vizier:J/ApJS/245/34/catalog',
                     max_distance=1 * u.arcsec,
                     colRA1='gaia_ra',
                     colDec1='gaia_dec',
                     colRA2='RAJ2000',
                     colDec2='DEJ2000')

type(table)
print(table)

# Now let's try the xmatch by querying the Vizier URLs for both catalogs
# table2 = XMatch.query(cat1='vizier:J/AJ/156/264/table2',
#                      cat2='vizier:J/ApJS/245/34/catalog',
#                      max_distance=1 * u.arcsec,
#                      colRA1='_RA',
#                      colDec1='_DEC',
#                      colRA2='RAJ2000',
#                      colDec2='DEJ2000')


# type(table2)
# print(table2)
# There are fewer stars when querying Vizier directly. Unclear why.


# In[5]:


table = table.to_pandas()
table.head(50)


# In[6]:


unq = np.unique(table['gaia_ra'], return_index=True, return_counts=True)

table['xmatch_count'] = np.zeros(len(table))
#table['xmatch_count'].iloc[unq[1]] = unq[2]
for i in range(len(table)):
    arg = unq[0] == table['gaia_ra'].iloc[i]
    table['xmatch_count'].iloc[i] = unq[2][arg]


# In[7]:


table = table.sort_values(by=["gaia_ra"])
table.head(20)


# In[8]:


table = table.merge(cks, how='right', left_on='gaia_ra', right_on='gaia_ra')
table.head()


# In[9]:


table = table.sort_values(["gaia_ra","angDist"], ascending = (True, True))
table.head()


# In[10]:


np.shape(table)


# In[11]:


table = table.drop_duplicates(subset="gaia_ra", keep="first")
np.shape(table)


# In[23]:


sns.set(font_scale=1.1, context="paper", style="ticks", palette="Blues_r")

fig, axes = plt.subplots(nrows=4,
                         ncols=4,
                         figsize=(12,10))

sc_kws = {"s":1, "color":"C1", "alpha":0.5, "rasterized":True}


prot = [table['d21_prot'],
        table['a18_period'],
        table['m15_Prot'],
        table['m13_Prot']]

for i,pr in enumerate(prot):
    axes[i][0].scatter(table['Teff'], prot[i], **sc_kws)
    axes[i][1].scatter(table['cks_Teff'], prot[i], **sc_kws)
    axes[i][2].scatter(table['bf18_Teff'], prot[i], **sc_kws)
    axes[i][3].scatter(table['m19_Teff'], prot[i], **sc_kws)
    
for i in range(4):
    for j in range(4):
        #if i<3:
        #    axes[i][j].set_xticklabels('')
        #if j>0:
        #    axes[i][j].set_yticklabels('')
            
        axes[i][j].set_xlim(7000,4100)
        axes[i][j].set_ylim(-2,65)

        
axes[3][0].set_xlabel('Effective temperature [K]\n(LAMOST; Xiang et al. 2019)')
axes[3][1].set_xlabel('Effective temperature [K]\n(CKS; Fulton & Petigura 2018)')
axes[3][2].set_xlabel('Effective temperature [K]\n(SPOCS; Brewer & Fischer 2018)')
axes[3][3].set_xlabel('Effective temperature [K]\n(Martinez et al. 2019)')

axes[0][0].set_ylabel('Rotation period [d]\n(David et al. 2021)')
axes[1][0].set_ylabel('Rotation period [d]\n(Angus et al. 2018)')
axes[2][0].set_ylabel('Rotation period [d]\n(Mazeh et al. 2015)')
axes[3][0].set_ylabel('Rotation period [d]\n(McQuillan et al. 2013)')

sns.despine()
plt.subplots_adjust(hspace=0.25,wspace=0.25)
plt.savefig('../figures/comparison.pdf')


# In[ ]:




