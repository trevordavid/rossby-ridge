from dustmaps.bayestar import BayestarWebQuery
from dustmaps.config import config
config.reset()

from dustmaps.bayestar import BayestarWebQuery
dustquery = BayestarWebQuery(version='bayestar2019')

import numpy as np
import astropy.units as units
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch

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

gyro_sequences = ['pleiades-ro', 'praesepe', 'ngc6811', 'ngc6819+ruprecht147']
gyro_ages = ['0.12 Gyr', '0.67 Gyr', '1 Gyr', '2.5 Gyr']

def gaia_edr3_vizier_xmatch(vizier_code):

    qcat = Vizier(columns=['_r'])
    qcat.ROW_LIMIT = -1

    cat1 = qcat.get_catalogs(vizier_code)[0]

    # Cross-match with Gaia EDR3
    search_radius = 1
    edr3 = XMatch.query(cat1=cat1,
                        cat2='vizier:I/350/gaiaedr3',
                        max_distance=search_radius * units.arcsec,
                        colRA1='_RA', colDec1='_DE')

    # "rename" some columns. Really, adding copies in case i forget the name, because it doesnt' matter here.
    edr3['RA'] = edr3['ra']
    edr3['Dec'] = edr3['dec']
    edr3['BP-RP'] = edr3['bp_rp']
    edr3['Gmag'] = edr3['phot_g_mean_mag']
    
    # Store in the object named "Amy"
    cat1 = edr3.copy()

    cmd_coeff = [-0.0319809, 4.08935, 5.76321, -6.98323, 3.06721, -0.589493, 0.0417076]
    hyades_cmd = np.poly1d(np.flip(cmd_coeff))

    #Query the dust map

    # I have McQuillan+2014 loaded in an astropy table called "Amy"
    distance_cat1 = 1e3/cat1['parallax']
    ifix = np.where(distance_cat1 <= 0)[0]
    if len(ifix) > 0:
        distance_cat1[ifix] = 1e4
    
    lg_coords = SkyCoord(cat1['ra']*units.deg,
                         cat1['dec']*units.deg,
                         distance=distance_cat1*units.pc,
                         frame='icrs')
    
    lg_gcoords = lg_coords.galactic
    map_coords = SkyCoord(lg_gcoords.l,
                          lg_gcoords.b,
                          distance=distance_cat1*units.pc,
                          frame='galactic')
    
    EKF = dustquery(map_coords, mode='median')

    # Set absolute G magnitude
    cat1['MG'] = cat1['phot_g_mean_mag'] - 5*np.log10(100/cat1['parallax'])
    # Fefine delta_CMD, the M_G mag deviation from the Hyades main sequence:
    dcmd = abs(cat1['MG'] - hyades_cmd(cat1['bp_rp']))
    # Do again assuming the dustmap gives a good reddening correction:
    dcmd_av = abs(cat1['MG'] - hyades_cmd(cat1['bp_rp']-0.415*EKF*2.74)-0.86*EKF*2.74)

    # Here, we use average extinction coefficients:
    # 0.415 *  A_V  = E(BP-RP)
    # 0.86 * A_V = A_G
    # And then to scale the Bayestar19 extinction value to A_V, we scale it by this: 
    # A_V = 2.74 * EKF (I can point you to their documentation about it)
    # Now, to trim the full sample, let's exclude astrometric binaries with RUWE < 1.2, distant stars with d < 1e3 pc,
    # photometric outliers (not even doing much of a cut on binaries, but mostly evolved or otherwise peculiar stars
    # by requesting that their MG be within 1 mag of the Hyades main sequence, using either no reddening/extinction or full Bayestar19 correction, and
    # finally select stars in low-reddening regions: 2.74*EKF < 0.2 mag, so A_V < 0.2.

    ikep = np.where((cat1['ruwe']<1.2) & (1e3/cat1['parallax']<1e3) & (EKF*2.7<0.2) & (np.isfinite(cat1['bp_rp'])) & (np.isfinite(cat1['MG'])) &
                ((dcmd<1) | (dcmd_av<1)))[0]

    return cat1, ikep


# Santos et al. 2021 cross-match with Gaia EDR3
san_vizier_code = "J/ApJS/255/17/table1"
san, san_ikep = gaia_edr3_vizier_xmatch(san_vizier_code)


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300

import seaborn as sns
sns.set(font_scale=1.3, context="paper", style="ticks")

def corsaro_tc(bprp):
  #Empirical relation between Gaia BP-RP color and convective turnover timescale from Corsaro et al. 2021
  #Calibrated using Kepler LEGACY asteroseismic sample
  b1_prime = -134.
  b2_prime = 341.7
  b3_prime = -150.6
  tau_c = b1_prime + b2_prime * bprp + b3_prime * bprp**2
  return tau_c

def constant_rossby(bprp, ro):
    #Return locus of rotation periods corresponding to constant Rossby number
    return ro * corsaro_tc(bprp)



####################################################################
# The Sun
sun = {"teff": 5772,
    "prot": 25.4,
    "e_prot": 25.4-24.5,
    "E_prot": 27-25.4,
    "bprp": 0.818,
    }
####################################################################

_bprp = np.linspace(0.55,0.97,1000)
_bprp_cluster = np.linspace(0.6,2,1000)

h_kws = {"bins":100, "cmap": "Blues", "density":True}
kde_kws = {"fill": True, "bw_adjust": 0.25, "cmap": "Blues"}
sun_kws = {"marker":"o", "color":"orange", "ms":8, "mfc":"None", "mew":1}


fig,(axes,axs) = plt.subplots(nrows=2,
                        ncols=3,
                        figsize=(12,6))


x, y = san['bp_rp'], san['Prot']
cb0 = axes[0].hist2d(x, y, **h_kws)
sns.kdeplot(
    x=x, 
    y=y, 
    ax=axs[0],
    **kde_kws
)
axes[0].text(0.75,0.07,str(len(x))+' stars', transform=axes[0].transAxes, size=9)

x, y = san['bp_rp'][san_ikep], san['Prot'][san_ikep]
cb1 = axes[1].hist2d(x, y, **h_kws)
sns.kdeplot(
    x=x, 
    y=y, 
    ax=axs[1],
    **kde_kws
)
axes[1].text(0.75,0.07,str(len(x))+' stars', transform=axes[1].transAxes, size=9)

x, y = san['bp_rp'][san_ikep]-0.1, san['Prot'][san_ikep]
cb2 = axes[2].hist2d(x, y, **h_kws)
sns.kdeplot(
    x=x, 
    y=y, 
    ax=axs[2],
    **kde_kws
)
axes[2].text(0.75,0.07,str(len(x))+' stars', transform=axes[2].transAxes, size=9)


for ax in zip(axes,axs):
    for i in range(2):
        ax[i].plot(sun["bprp"], sun["prot"], **sun_kws)
        ax[i].plot(sun["bprp"], sun["prot"], '.', color='orange')
        ax[i].set_xlim(0.5,1.3)
        ax[i].set_ylim(0,50)
        ax[i].set_xlabel(r"$G_\mathregular{BP}-G_\mathregular{RP}$ [mag]")
        ax[i].set_ylabel("Rotation period [d]")
        ax[i].plot(_bprp, constant_rossby(_bprp, 1.2*0.496), color='orange', ls='-', label=r'$\mathregular{Ro} = 1.2 \mathregular{R}_\odot$', zorder=np.inf)
        ax[i].plot(_bprp, constant_rossby(_bprp, 0.496), color='orange', ls='--', label=r'$\mathregular{Ro} = 1.0 \mathregular{R}_\odot$', zorder=np.inf)
        ax[i].plot(_bprp, constant_rossby(_bprp, 0.8*0.496), color='orange', ls=':', label=r'$\mathregular{Ro} = 0.8 \mathregular{R}_\odot$', zorder=np.inf)
        ax[i].legend(prop={"size":8})
        
        #for j,seq in enumerate(gyro_sequences):
        #    ax[i].plot(_bprp_cluster, curtis_gyrochrone(_bprp_cluster, kind=seq), label=gyro_ages[j], color='grey', lw=3, alpha=0.4)

for i in range(3):
    for j,seq in enumerate(gyro_sequences):
        if i==2:
            axes[i].plot(_bprp_cluster-0.1, curtis_gyrochrone(_bprp_cluster, kind=seq), label=gyro_ages[j], color='grey', lw=3, alpha=0.4)
            axs[i].plot(_bprp_cluster-0.1, curtis_gyrochrone(_bprp_cluster, kind=seq), label=gyro_ages[j], color='grey', lw=3, alpha=0.4)
        else:
            axes[i].plot(_bprp_cluster, curtis_gyrochrone(_bprp_cluster, kind=seq), label=gyro_ages[j], color='grey', lw=3, alpha=0.4)
            axs[i].plot(_bprp_cluster, curtis_gyrochrone(_bprp_cluster, kind=seq), label=gyro_ages[j], color='grey', lw=3, alpha=0.4)        

axes[0].set_title("All stars\n(Santos et al. 2021 + Gaia EDR3)")
axes[1].set_title("Low reddening sample\n(A$_\mathregular{V}$<0.2 mag)")
axes[2].set_title("Low reddening sample\nwith -0.1 mag shift")
plt.tight_layout()
plt.savefig('../figures/gaia-santos.pdf')

