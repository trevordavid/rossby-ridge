import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.table import Table
import pandas as pd
import seaborn as sns

mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300

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

h_kws = {"bins":400, "cmap": "Blues", "cmin": 0, "cmax":5, "density":False}

phot_kws = {"ms": 0.1,
            "color": "orange",
            "alpha": 1,
            "rasterized":True}

fig,axes = plt.subplots(nrows=1,ncols=2,
                      figsize=(10,4))

for i in range(2):
    cb1 = axes[i].hist2d(bprp[arg], MG[arg], **h_kws)
    axes[i].set_xlim(-0.5,4)
    axes[i].set_ylim(13,-3)
    axes[i].set_xlabel(r'G$_\mathregular{BP}$-G$_\mathregular{RP}$ [mag]')
    axes[i].set_ylabel(r'M$_\mathregular{G}$ [mag]')

axes[0].plot(bprp[mcq_rot], MG[mcq_rot], '.', label='photometric periods\n(McQuillan et al. 2014)', **phot_kws)
axes[1].plot(bprp[san_rot], MG[san_rot], '.', label='photometric periods\n(Santos et al. 2021)', **phot_kws)

for i in range(2):        
    
    axes[i].plot(cks['gaia_bp_rp'][ridge], cks['MG'][ridge], 'k.', ms=1, label='long-period pile-up (this work)', zorder=999)
    sns.kdeplot(bprp[hall_ast], MG[hall_ast], levels=4, color='k', ax=axes[i], **{"linewidths":0.75}, zorder=998, label='asteroseismic periods\n(Hall et al. 2021)')
        
    lgnd = axes[i].legend(loc='lower left', prop={'size':8})
    #change the marker size manually for both lines
    #lgnd.legendHandles[0]._legmarker.set_markersize(6)
    #lgnd.legendHandles[1]._legmarker.set_markersize(6)
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    
plt.savefig('../figures/cmd.pdf')