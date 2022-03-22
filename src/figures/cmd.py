import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.table import Table
import pandas as pd
import seaborn as sns
import scipy.stats as st

import matplotlib.colors as colors

mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["legend.markerscale"] = 10

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('Greys_r')
new_cmap = truncate_colormap(cmap, 0.2, 0.8)

sns.set(
        context="paper",
        style="ticks",
        font_scale=1.2,
        palette="colorblind"
        )

#Gaia-Kepler cross-match from Megan Bedell
dr2 = pd.read_parquet('../data/kepler_dr2_1arcsec.parquet')
dr2 = dr2.rename(columns={"logg": "KIC_logg"})
dr2 = dr2.add_prefix("dr2_")
dr2['MG'] = dr2['dr2_phot_g_mean_mag'] - 5 * (np.log10(dr2['dr2_r_est']) - 1)

#McQuillan et al. 2014
mcq = pd.read_parquet('../data/mcquillan2014_table1.parquet')

#Santos et al. 2021
san = pd.read_csv('../data/S21_rotators.csv')
san = san.add_prefix('san_')

#Hall et al. 2021
hall = Table.read("https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/table1.dat",
                  readme="https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/ReadMe",
                  format="ascii.cds")

hall = hall.to_pandas()
hall = hall.add_prefix('hall_')

#CKS (Fulton & Petigura 2018)
cks = pd.read_parquet('../data/cks_merged.parquet')
# The dataframe has a row entry for each KOI, meaning individual star are represented N times
# where N is the number of KOIs detected around that star so we drop duplicates.
cks = cks.drop_duplicates(subset=['kepid'], keep='first')

#Merge Gaia-DR2 catalog with others
dr2 = dr2.merge(mcq, left_on='dr2_kepid', right_on='mcq_KIC', how='left')
dr2 = dr2.merge(san, left_on='dr2_kepid', right_on='san_KIC', how='left')
dr2 = dr2.merge(hall, left_on='dr2_kepid', right_on='hall_KIC', how='left')

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

mask = (cks['p20_cks_slogg']>4) #main sequence
ridge = (cks['cks_Teff']>5850)
ridge &= (cks['cks_Teff']<6500)
ridge &= (cks['d21_prot']<ridge_hi(cks['cks_Teff']))
ridge &= (cks['d21_prot']>ridge_lo(cks['cks_Teff']))
ridge &= mask


cks['MG'] = cks['gaia_phot_g_mean_mag'] - 5 * (np.log10(cks['gaia_r_est']) - 1)


mcq_rot = np.isfinite(dr2['mcq_Prot'])
san_rot = np.isfinite(dr2['san_Prot'])
hall_ast = np.isfinite(dr2['hall_P'])

bprp = dr2['dr2_bp_rp']
MG   = dr2['MG']
arg  = np.isfinite(bprp) & np.isfinite(MG)

h_kws = {"bins":200, "cmap": new_cmap, "cmin": 1, "cmax":100, "density":False}

phot_kws = {"ms": 0.1,
            "color": "orange",
            "alpha": 1,
            "rasterized":True}

fig,axes = plt.subplots(nrows=1,ncols=3,
                      figsize=(14,4))

for i in range(3):
    cb1 = axes[i].hist2d(bprp[arg], MG[arg], **h_kws)
    axes[i].set_xlim(-0.5,4)
    axes[i].set_ylim(13,-3)
    axes[i].set_xlabel(r'G$_\mathregular{BP}$-G$_\mathregular{RP}$ [mag]')
    axes[i].set_ylabel(r'M$_\mathregular{G}$ [mag]')

#axes[0].plot(bprp[mcq_rot], MG[mcq_rot], ',', **phot_kws)
axes[1].plot(bprp[san_rot], MG[san_rot], ',', **phot_kws)
axes[2].plot(bprp[san_rot], MG[san_rot], ',', **phot_kws)

#Hack for legend troubles
#axes[0].plot(bprp[mcq_rot].iloc[0], MG[mcq_rot].iloc[0], '.', ms=1, color='orange', alpha=1, label='photometric periods\n(McQuillan et al. 2014)')
axes[1].plot(bprp[san_rot].iloc[0], MG[san_rot].iloc[0], '.', ms=1, color='orange', alpha=1, label='photometric periods\n(Santos et al. 2021)')
axes[2].plot(bprp[san_rot].iloc[0], MG[san_rot].iloc[0], '.', ms=1, color='orange', alpha=1, label='photometric periods\n(Santos et al. 2021)')

axes[1].legend(loc='lower left', prop={'size':8})

for i in range(2,3):        
    
    axes[i].plot(cks['gaia_bp_rp'][ridge], cks['MG'][ridge], '.', color='C0', ms=1, label='long-period pile-up (this work)', zorder=999)
   
    #Represent the asteroseismic sample with Guassian KDE contours
    x, y = bprp[hall_ast], MG[hall_ast]

    xmin, xmax = 0.8*x.min(),1.2*x.max()
    ymin, ymax = 0.8*y.min(),1.2*y.max()

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    cset = axes[i].contour(xx, yy, f, colors='C0', levels=4, linewidths=0.5, zorder=998)
    #cset.collections[0].set_label('asteroseismic periods\n(Hall et al. 2021)')

    #Hack for asteroseismic legend label
    axes[i].plot([-100,-99],[12,12],'k',label='asteroseismic periods\n(Hall et al. 2021)')

    lgnd = axes[i].legend(loc='lower left', prop={'size':8})
    
sns.despine()
plt.savefig('../figures/cmd.pdf')