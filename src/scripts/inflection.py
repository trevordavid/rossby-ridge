import paths
import numpy as np
import pandas as pd
import emcee
from scipy.optimize import curve_fit

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import seaborn as sns 

mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300

#CKS (Fulton & Petigura 2018)
cks = pd.read_parquet('../data/cks_merged.parquet')

#LAMOST-McQuillan
lam = pd.read_csv('../data/kepler_lamost.csv')
lam = lam.drop_duplicates(subset=['KIC'], keep='first')

def ridge_hi(teff):
    m = (2-24)/(6500-5800)
    b = (2 - m*6500) 
    return m*teff + b

def ridge_lo(teff):
    m = (2-24)/(6500-5800)
    b = (-5 - m*6500) 
    return m*teff + b

mask = (cks['p20_cks_slogg']>4) #main sequence
ridge = (cks['p20_cks_steff']>5850)
ridge &= (cks['p20_cks_steff']<6500)
ridge &= (cks['d21_prot']<ridge_hi(cks['p20_cks_steff']))
ridge &= (cks['d21_prot']>ridge_lo(cks['p20_cks_steff']))
ridge &= mask

# Parallelogram
xp = [6500,6500,5850,5850]
yp = [ridge_lo(6500),ridge_hi(6500),ridge_hi(5850),ridge_lo(5850)]

# Select stars on the pile-up
x, y = np.array(cks['cks_Teff'][ridge]), np.array(cks['d21_prot'][ridge])
yerr = np.ones(len(y))


def piecewise_linear(x, x0, y0, k1, k2):
    y = np.piecewise(x, [x < x0, x >= x0],
                     [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    return y

def log_likelihood(theta, x, y, yerr):
    x0, y0, k1, k2, f = theta
    model = piecewise_linear(x, x0, y0, k1, k2)
    sigma2 = yerr ** 2 + model ** 2 * f ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    x0, y0, k1, k2, f = theta
    if 6150 < x0 < 6350 and 0 < y0 < 10 and -1 < k1 < 1 and -1 < k2 < 1 and 0 < f < 10:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


popt_piecewise, pcov = curve_fit(piecewise_linear, x, y, p0=[6250, 7,5 -0.025, 0.025])
popt_piecewise = np.append(popt_piecewise, 0.1)

pos = popt_piecewise + 1e-4 * np.random.randn(32, len(popt_piecewise))
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(x, y, yerr)
)
sampler.run_mcmc(pos, 5000, progress=True);


tau = sampler.get_autocorr_time()
print(tau)
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(len(flat_samples)/tau)

for j in range(np.shape(flat_samples)[1]):
    print(j, np.median(flat_samples[:,j]), np.std(flat_samples[:,j]))

#Stars with large temperature differences between CKS (Fulton & Petigura 2018) & SPOCS (Brewer & Fischer 2018)
arg = (abs(cks['bf18_Teff']-cks['cks_Teff'])>100)

sun = {"teff": 5772,
       "prot": 25.4,
       "e_prot": 25.4-24.5,
       "E_prot": 27-25.4
      }


sns.set(font_scale=1.3, context="paper", style="ticks")


fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5))

ax.plot(lam['Teff_lam'], lam['Prot'], 'o', ms=4, alpha=0.4, color='lightgrey', label='LAMOST–McQuillan', mew=0, rasterized=True)
ax.plot(cks['cks_Teff'], cks['d21_prot'], 'o', ms=2, label='CKS–David', rasterized=True)
ax.plot(cks['cks_Teff'][arg], cks['d21_prot'][arg], 'o', ms=5, mfc='None', mew=0.5, color='k')

ax.errorbar(5200,1,xerr=np.nanmedian(cks['p20_cks_steff_err1']),yerr=0,fmt='.',color='k', zorder=999)
ax.errorbar(5200,3,xerr=np.nanmedian(lam['e_Teff_lam'][lam['Teff_lam']>5800]),yerr=0,fmt='.',color='k', zorder=999)
ax.set_xlim(6500,5600)
ax.set_ylim(-1,30)
ax.add_patch(patches.Polygon(xy=list(zip(xp,yp)), fill=False, lw=0.5, color='k', zorder=np.inf))

ax.errorbar(sun["teff"], sun["prot"], yerr=np.vstack([sun["e_prot"], sun["E_prot"]]), fmt="o", 
                   color="C1", mec="white", ms=6, label='Sun')

leg = plt.legend(prop={"size":12})

# Plot some fits
inds = np.random.randint(len(flat_samples), size=50)
xtmp = np.linspace(5850,6500,100)

for ind in inds:
    sample = flat_samples[ind]
    ax.plot(xtmp, piecewise_linear(xtmp, sample[0], sample[1], sample[2], sample[3]), "C0", alpha=0.1)
    
ax.set_xlabel('Effective temperature [K]')
ax.set_ylabel('Rotation period [d]')
plt.savefig('../figures/inflection.pdf')
plt.show()