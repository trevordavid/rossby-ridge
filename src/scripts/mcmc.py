#!/usr/bin/env python
# coding: utf-8

import paths
import numpy as np
import pandas as pd

from astropy.table import Table
import emcee
import corner
from scipy.optimize import curve_fit
from multiprocessing import Pool

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300

import seaborn as sns

sns.set(
        context="paper",
        style="ticks",
        font_scale=1.2,
        palette="colorblind"
        )


######################################################################################
#McQuillan et al. 2013
mcq_koi = Table.read("https://cdsarc.cds.unistra.fr/ftp/J/ApJ/775/L11/table1.dat",
                readme="https://cdsarc.cds.unistra.fr/ftp/J/ApJ/775/L11/ReadMe",
                format="ascii.cds")
mcq_koi = mcq_koi.to_pandas()
mcq_koi = mcq_koi.add_prefix('mcq_')


#McQuillan et al. 2014
mcq = pd.read_parquet(paths.data / 'mcquillan2014_table1.parquet')
######################################################################################


######################################################################################
# California-Kepler Survey (Fulton & Petigura 2018)
# This data table has been augmented with data from other surveys (see David et al. 2021)
cks = pd.read_parquet(paths.data / 'cks_merged.parquet')
# The dataframe has a row entry for each KOI, meaning individual star are represented N times
# where N is the number of KOIs detected around that star so we drop duplicates.
cks = cks.drop_duplicates(subset=['kepid'], keep='first')
cks = cks.merge(mcq_koi, how='left', left_on='kepid', right_on='mcq_KIC')
######################################################################################


######################################################################################
# LAMOST-Kepler 
lam = pd.read_parquet(paths.data / 'kepler_lamost.parquet')
#print('LAMOST unique KIC targets:', len(np.unique(lam["KIC"])))
#print('LAMOST unique DR2 targets:', len(np.unique(lam["DR2Name"])))

# Drop duplicate sources, keeping the one with the brighter G magnitude
lam = lam.sort_values(["KIC", "Gmag"], ascending = (True, True))
lam = lam.merge(mcq, how='left', left_on="KIC", right_on="mcq_KIC")
lam = lam.drop_duplicates(subset=['KIC'], keep='first')

lam_mask = (lam["Teff_lam"]>3000)
lam_mask = (lam["Teff_lam"]<8000)
lam_mask &= (lam["logg_lam"]>4)
lam_mask &= (lam["logg_lam"]<5)
lam_mask &= (abs(lam["feh_lam"])<2)
lam = lam[lam_mask]
######################################################################################


######################################################################################
def convective_turnover_timescale(teff):    
    #Returns convective turnover timescale in days    
    #Gunn et al. 1998 relation, from Cranmer & Saar 2011
    return 314.24*np.exp( -(teff/1952.5) - (teff/6250.)**18. ) + 0.002
    
def constant_rossby(teff, ro):
    #Return locus of rotation periods corresponding to constant Rossby number
    return ro * convective_turnover_timescale(teff)

lam["Ro"] = lam["Prot"]/convective_turnover_timescale(lam["Teff_lam"])    
######################################################################################

    
def percentile_bootstrap(nsamples=100, 
                         f=0.5, 
                         pctl=90.):

    # nsamples : number of bootstrap resamplings to perform
    # f        : fraction of data to leave out in each bin
    # pctl     : percentile to compute
    
    _teff = np.linspace(4000,7500,1000)
    teff_bin_centers = np.arange(4000,7020,20)
    
    per_pctl = np.zeros(len(teff_bin_centers))
    per_pctl_err = np.zeros(len(teff_bin_centers))
    
    for i, tc in enumerate(teff_bin_centers):
        arg = (abs(lam["Teff_lam"]-tc)<100) & (lam["Ro"]<5/3)
        pctl_arr = []
        
        for n in range(nsamples):
            _x = np.array(lam["Prot"][arg])
            pctl_arr.append(np.nanpercentile(_x[np.random.choice(len(_x), int(f*len(_x)), replace=True)], pctl))
        
        per_pctl[i] = np.mean(pctl_arr)
        per_pctl_err[i] = np.std(pctl_arr)    
        
        
    return teff_bin_centers, per_pctl, per_pctl_err
     
teff_bin_centers, period_90th_pctl, e_period_90th_pctl = percentile_bootstrap(pctl=90.)
teff_bin_centers, period_10th_pctl, e_period_10th_pctl = percentile_bootstrap(pctl=10.)    


############################################
cks_teff = cks["p20_cks_steff"]
cks_e_teff = cks["p20_cks_steff_err1"]
cks_prot = cks["d21_prot"]

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
############################################


lam_teff = lam["Teff_lam"]
lam_e_teff = lam["e_Teff_lam"]
lam_prot = lam["Prot"]

hall = Table.read("https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/table1.dat",
                  readme="https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/ReadMe",
                  format="ascii.cds")

x_lam = np.ascontiguousarray(teff_bin_centers, dtype=np.float64)
y_lam = np.ascontiguousarray(period_90th_pctl, dtype=np.float64)

x_cks = np.ascontiguousarray(cks_teff[ridge], dtype=np.float64)
y_cks = np.ascontiguousarray(cks_prot[ridge], dtype=np.float64)

hall_ms = (hall["Type"] == "MS") #& (hall["Teff"]>5800)
x_hal = np.ascontiguousarray(hall["Teff"][hall_ms], dtype=np.float64) 
y_hal = np.ascontiguousarray(hall["P"][hall_ms],  dtype=np.float64)


def constant_rossby_sampler(x, y, yerr,
                            teff_min=5000, 
                            teff_max=6250,
                            ndraws=5000, 
                            trace_plot=True,
                            corner_plot=True):
    
    m = (np.isfinite(x)) & (np.isfinite(y)) & (x>teff_min) & (x<teff_max)
    x = x[m]
    y = y[m]
    yerr = yerr[m]

    #Non-linear least squares fit
    popt, pcov = curve_fit(constant_rossby, x, y)

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=yerr, color='k', fmt='.')
    ax.plot(x, constant_rossby(x, *popt), 'C1')
    ax.set_xlabel("Teff [K]")
    ax.set_ylabel("Prot [d]")
    ax.invert_xaxis()
    #plt.show()
    
    #Functions for MCMC sampling
    def log_likelihood(theta, x, y, yerr):
        ro, f = theta
        model = constant_rossby(x, ro)
        sigma2 = yerr ** 2 + model ** 2 * f ** 2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

    def log_prior(theta):
        ro, f = theta
        if 0.1 < ro < 10 and 0 < f < 10:
            return 0.0
        return -np.inf

    def log_probability(theta, x, y, yerr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y, yerr)


    np.random.seed(20211018)
    
    pos = [popt[0],0.1] + 1e-4 * np.random.randn(32, 2)
    nwalkers, ndim = pos.shape
    
    import time
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(x, y, yerr)
    )        
    start = time.time()
    sampler.run_mcmc(pos, ndraws, progress=True)    
    end = time.time()
    serial_time = end - start
    print("Sampling took {0:.1f} seconds".format(serial_time))
    
    
    if trace_plot==True:
        fig, axes = plt.subplots(ndim, figsize=(5, 3.5), sharex=True)
        samples = sampler.get_chain()
        labels = ["Ro", "f"]

        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])

        axes[-1].set_xlabel("Step number");

        tau = sampler.get_autocorr_time()
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        print(flat_samples.shape)
        print('Autocorrelation length:', tau)
        print('Chain length / autocorrelation length:', len(flat_samples)/tau)

        for i in range(flat_samples.shape[1]):
            print(np.mean(flat_samples[:,i]), np.std(flat_samples[:,i]))        
            #print(np.median(flat_samples[:,i]), np.percentile(flat_samples[:,i],[16,84])-np.median(flat_samples[:,i]))        
    
    
    if corner_plot==True:        

        fig = corner.corner(
            flat_samples, labels=labels,
        );

    
    return flat_samples


x1 = teff_bin_centers
y1 = period_90th_pctl
y1err = np.ones(len(y1))
arg1 = (x1>5800) & (x1<6250)
x1 = x1[arg1]
y1 = y1[arg1]
y1err = y1err[arg1]


flat_samples_90 = constant_rossby_sampler(x = x1,
                                         y = y1,
                                         yerr = y1err,
                                         teff_min = 5000,
                                         teff_max = 6250)

flat_samples_10 = constant_rossby_sampler(x = teff_bin_centers,
                                         y = period_10th_pctl,
                                         yerr = e_period_10th_pctl,
                                         teff_min = 5000,
                                         teff_max = 6250)


def rossby_plot(teff_bin_centers, flat_samples_10, flat_samples_90):
    
    y10 = period_10th_pctl
    y90 = period_90th_pctl
    
    yerr10 = e_period_10th_pctl
    yerr90 = e_period_90th_pctl
    
    ro10 = np.median(flat_samples_10[:,0])
    ro90 = np.median(flat_samples_90[:,0])
    
    model_10 = constant_rossby(teff_bin_centers, ro10)
    frac_resid_10 = 100*(y10-model_10)/y10
    
    fmed_10 = np.median(flat_samples_10[:,1])
    sigma_10 = np.sqrt(yerr10**2 + model_10**2 * fmed_10**2)
    
    model_90 = constant_rossby(teff_bin_centers, ro90)
    frac_resid_90 = 100*(y90-model_90)/y90
    
    fmed_90 = np.median(flat_samples_90[:,1])
    sigma_90 = np.sqrt(yerr90**2 + model_90**2 * fmed_90**2)    
        
    sns.set(font_scale=1.4, context="paper", style="ticks")
    sns.set_palette("Blues")

    teff_bin_centers = np.arange(4000,7020,20)  

    fig = plt.figure(figsize=(6,8))
    ax1 = plt.subplot2grid((5, 3), (0, 0), colspan=3, rowspan=2)
    ax2 = plt.subplot2grid((5, 3), (2, 0), colspan=3, rowspan=2)
    ax3 = plt.subplot2grid((5, 3), (4, 0), colspan=3, rowspan=1)


    sns.kdeplot(
        x=lam["Teff_lam"], 
        y=lam["Prot"], 
        fill=True, 
        bw_adjust=0.5,
        ax=ax1
    )

    ax1.errorbar(teff_bin_centers, period_90th_pctl, yerr=e_period_90th_pctl, fmt='.', color='k', ms=6, alpha=0.9, label='90th percentile')
    ax1.errorbar(teff_bin_centers, period_10th_pctl, yerr=e_period_10th_pctl, fmt='.', color='k', mfc='white', ms=6, alpha=0.9, label='10th percentile', lw=0.25)
    ax1.errorbar(teff_bin_centers, period_10th_pctl, yerr=e_period_10th_pctl, fmt='.', color='white', mfc='white', ms=2, alpha=0.9, lw=0)
    ax1.set_ylim(0,40)
    ax1.legend()

    _teff = np.linspace(4000,7000,1000)


    for i,_ro in enumerate([3,ro90,0.75,ro10]):
        ax2.plot(_teff, constant_rossby(_teff, _ro), label='Ro = '+"{:.2f}".format(_ro), lw=3, alpha=1)


    ax2.errorbar(teff_bin_centers, period_90th_pctl, yerr=e_period_90th_pctl, fmt='.', color='k', ms=6, alpha=0.9)
    ax2.errorbar(teff_bin_centers, period_10th_pctl, yerr=e_period_10th_pctl, fmt='.', color='k', mfc='white', ms=6, alpha=0.9, lw=0.25)
    ax2.errorbar(teff_bin_centers, period_10th_pctl, yerr=e_period_10th_pctl, fmt='.', color='white', mfc='white', ms=2, alpha=0.9, lw=0)
    
    ax2.semilogy()
    ax2.set_ylim(0.1,100)
    ax2.set_yticks([0.1,1,10,100])
    ax2.set_yticklabels(['0.1','1','10','100'])
    ax2.legend()

    for ax in [ax1,ax2]:
        ax.set_xlabel("Effective temperature [K]")
        ax.set_ylabel("Rotation period [d]")
        ax.set_xlim(7000,4500)   

    ax3.errorbar(teff_bin_centers, frac_resid_90, yerr=sigma_90, fmt='.', color='k', ms=6)
    ax3.errorbar(teff_bin_centers, frac_resid_10, yerr=sigma_10, fmt='.', color='k', mfc='white', ms=6)            
    ax3.set_xlim(7000,4500)
    ax3.set_ylim(-50,50)
    ax3.axhline(0, color='k', ls='--')
    ax3.set_xlabel("Effective temperature [K]")
    ax3.set_ylabel("Residuals [%]")

    #sns.despine()
    plt.tight_layout()
    plt.savefig(paths.figures / 'mcmc.pdf')
    
    return

rossby_plot(teff_bin_centers, flat_samples_10, flat_samples_90)



