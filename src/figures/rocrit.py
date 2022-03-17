import numpy as np
import pandas as pd

from astropy.table import Table

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
mcq = pd.read_parquet('../data/mcquillan2014_table1.parquet')
######################################################################################


######################################################################################
# California-Kepler Survey (Fulton & Petigura 2018)
# This data table has been augmented with data from other surveys (see David et al. 2021)
cks = pd.read_parquet('../data/cks_merged.parquet')
# The dataframe has a row entry for each KOI, meaning individual star are represented N times
# where N is the number of KOIs detected around that star so we drop duplicates.
cks = cks.drop_duplicates(subset=['kepid'], keep='first')
cks = cks.merge(mcq_koi, how='left', left_on='kepid', right_on='mcq_KIC')
######################################################################################


######################################################################################
# LAMOST-Kepler 
lam = pd.read_csv('../data/kepler_lamost.csv')
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

def rocrit_teff_shift(teff, teff_shift):
    #Return locus of rotation periods corresponding to constant Rossby number
    return 2 * convective_turnover_timescale(teff+teff_shift)

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

mask = (cks['p20_cks_slogg']>4.1) #main sequence
ridge = (cks['p20_cks_steff']>5850)
ridge &= (cks['p20_cks_steff']<6500)
ridge &= (cks['d21_prot']<ridge_hi(cks['p20_cks_steff']))
ridge &= (cks['d21_prot']>ridge_lo(cks['p20_cks_steff']))
ridge &= mask
############################################


#Detrending LAMOST Teff
def lamost_teff_detrend(teff):   
    dteff = 2.55513439e-13*teff**5 - 7.18129973e-09*teff**4 + 8.04175914e-05*teff**3 - 4.48417848e-01*teff**2 + 1.24490338e+03*teff - 1.37649898e+06
    return teff-dteff

lam["Teff_lam"] = lamost_teff_detrend(lam["Teff_lam"])

lam_teff = lam["Teff_lam"]
lam_e_teff = lam["e_Teff_lam"]
lam_prot = lam["Prot"]

#Detrended!
teff_bin_centers2, period_90th_pctl2, e_period_90th_pctl2 = percentile_bootstrap(pctl=90.)
teff_bin_centers2, period_10th_pctl2, e_period_10th_pctl2 = percentile_bootstrap(pctl=10.)    

hall = Table.read("https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/table1.dat",
                  readme="https://cdsarc.cds.unistra.fr/ftp/J/other/NatAs/5.707/ReadMe",
                  format="ascii.cds")

m_lam = (np.isfinite(period_90th_pctl)) & (teff_bin_centers>5800) & (teff_bin_centers<6250)
x_lam = np.ascontiguousarray(teff_bin_centers[m_lam], dtype=np.float64)
y_lam = np.ascontiguousarray(period_90th_pctl[m_lam], dtype=np.float64)

m_lam_det = (np.isfinite(period_90th_pctl2)) & (teff_bin_centers2>5800) & (teff_bin_centers2<6250)
x_lam_det = np.ascontiguousarray(teff_bin_centers2[m_lam], dtype=np.float64)
y_lam_det = np.ascontiguousarray(period_90th_pctl2[m_lam], dtype=np.float64)

x_cks = np.ascontiguousarray(cks_teff[ridge], dtype=np.float64)
y_cks = np.ascontiguousarray(cks_prot[ridge], dtype=np.float64)

hall_ms = (hall["Type"] == "MS") #& (hall["Teff"]>5800)
x_hal = np.ascontiguousarray(hall["Teff"][hall_ms], dtype=np.float64) 
y_hal = np.ascontiguousarray(hall["P"][hall_ms],  dtype=np.float64)


sns.set_palette("Blues")

from scipy.optimize import curve_fit

def lsq_fit_fixed_rocrit(x, y):
    
    #Non-linear least squares fit
    popt, pcov = curve_fit(rocrit_teff_shift, x, y)
    model_teff = np.linspace(5000,7000)
    
    return popt, rocrit_teff_shift(model_teff, *popt)

teff_labels = ['CKS', 'SPOCS', 'M19']
ebar_kws = {"fmt":'.', "color":'k', "ms":2, "lw":0.5}
leg_kws = {"loc": 'upper left'}

fig, axes = plt.subplots(nrows=1,ncols=4,figsize=(12*1.1,3*1.1))

for i,teff in enumerate([cks['cks_Teff'],
                         cks['bf18_Teff'],
                         cks['m19_Teff']]):
    
                        
    model_teff = np.linspace(5000,7000)
    
    _xdata = teff[ridge]
    _ydata = cks['d21_prot'][ridge]
    _arg = (np.isfinite(_xdata)) & (np.isfinite(_ydata)) & (_xdata<6250)
    
    #First fit the fixed Rocrit = 2 model
    _popt, _pcov = curve_fit(rocrit_teff_shift, _xdata[_arg], _ydata[_arg])
    _model = rocrit_teff_shift(model_teff, _popt)
    
    #Next fit the variable Rocrit model
    __popt, __pcov = curve_fit(constant_rossby, _xdata[_arg], _ydata[_arg])
    __model = constant_rossby(model_teff, __popt) 
    
    expected_fix = rocrit_teff_shift(_xdata[_arg], _popt)
    expected_var = constant_rossby(_xdata[_arg], __popt)
    
    chisq_fix = np.sum((_ydata[_arg] - expected_fix)**2 / expected_fix)
    chisq_var = np.sum((_ydata[_arg] - expected_var)**2 / expected_var)
    
    #bic_fix = len(_ydata[_arg]) - 2*(-0.5*chisq_fix)
    #bic_var = len(_ydata[_arg]) - 2*(-0.5*chisq_var)

    axes[i].errorbar(_xdata[_arg], _ydata[_arg], yerr = 0.1*_ydata[_arg], 
                     **ebar_kws)
    axes[i].plot(model_teff, _model, color='orange', lw=3, alpha=0.5,
                 label=r'Ro$_\mathregular{crit}$ = 2 ($\mathregular{\Delta}$T$_\mathregular{eff}$ = '+str(int(_popt[0]))+')')
    axes[i].plot(model_teff, __model, color='C2', lw=3, alpha=0.5,
                 label='Ro$_\mathregular{crit}$ = '+str(np.round(__popt[0],2))+' ($\mathregular{\Delta}$T$_\mathregular{eff}$ = 0)')
    axes[i].set_xlim(6400,5600)
    axes[i].set_ylim(-2,30)
    axes[i].text(0.05,0.05, 
                 r'$\mathregular{\Delta \chi^2} = $' + str(np.round(chisq_fix-chisq_var,2)),
                 transform=axes[i].transAxes)
    axes[i].legend(**leg_kws)
    axes[i].set_xlabel(teff_labels[i]+' Effective temperature [K]')
    axes[i].set_ylabel('Rotation period [d]')

#First fit the fixed Rocrit = 2 model
_popt, _model = lsq_fit_fixed_rocrit(x = x_lam_det,
                                     y = y_lam_det)

#Next fit the variable Rocrit model
__popt, __pcov = curve_fit(constant_rossby, x_lam_det, y_lam_det)
__model = constant_rossby(model_teff, __popt) 

expected_fix = rocrit_teff_shift(x_lam_det, _popt)
expected_var = constant_rossby(x_lam_det, __popt)

chisq_fix = np.sum((y_lam_det - expected_fix)**2 / expected_fix)
chisq_var = np.sum((y_lam_det - expected_var)**2 / expected_var)

axes[3].errorbar(x_lam_det, y_lam_det, yerr = 0.1*y_lam_det, 
                 **ebar_kws)

axes[3].plot(model_teff, _model, color='orange', lw=3, alpha=0.5,
             label=r'Ro$_\mathregular{crit}$ = 2 ($\mathregular{\Delta}$T$_\mathregular{eff}$ = '+str(int(_popt[0]))+')')
axes[3].plot(model_teff, __model, color='C2', lw=3, alpha=0.5,
             label='Ro$_\mathregular{crit}$ = '+str(np.round(__popt[0],2))+' ($\mathregular{\Delta}$T$_\mathregular{eff}$ = 0)')

axes[3].text(0.05,0.05, 
             r'$\mathregular{\Delta \chi^2} = $' + str(np.round(chisq_fix-chisq_var,2)),
             transform=axes[3].transAxes)

axes[3].set_xlim(6400,5600)
axes[3].set_ylim(-2,30)
axes[3].legend(**leg_kws)
axes[3].set_xlabel('LAMOST Effective temperature [K]')
axes[3].set_ylabel('Rotation period [d]')

sns.despine()
plt.tight_layout()
plt.savefig('../figures/rocrit.pdf')