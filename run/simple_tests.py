'''

run spectral fitter through very simple test 


'''
import os 
import h5py
import fsps
import numpy as np
from astropy import units as U 
from astropy.cosmology import FlatLambdaCDM
from firefly_fitter import fitter
# --- fomospec ---
from fomospec import util as UT 
from fomospec import spectra as Spec
from fomospec import fitters as Fitters
# -- plotting --
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False


age_bounds = np.array(((0.000,0.0025,0.005),(0.005,0.010,0.015),(0.015,0.020,0.025),(0.025,0.030,0.035),(0.035,0.0400,0.045),(0.045,0.050,0.055),(0.055,0.060,0.065),(0.065,0.070,0.075),(0.075,0.0800,0.085),(0.085,0.090,0.095),(0.095,0.100,0.125),(0.125,0.150,0.175),(0.175,0.20,0.225),(0.225,0.250,0.275),(0.275,0.300,0.325),(0.325,0.350,0.375),(0.375,0.40,0.425),(0.425,0.450,0.475),(0.475,0.5125,0.55),(0.550,0.600,0.650),(0.650,0.70,0.750),(0.75,0.80,0.85),(0.85,0.90,0.95),(0.95,1.0375,1.125),(1.125,1.25,1.375),(1.375,1.50,1.625),(1.625,1.75,1.875),(1.875,2.00,2.125),(2.125,2.25,2.375),(2.375,2.50,2.625),(2.625,2.75,2.875),(2.875,3.00,3.125),(3.125,3.25,3.375),(3.375,3.50,3.625),(3.625,3.75,3.875),(3.875,4.00,4.25),(4.25,4.50,4.75),(4.75,5.00,5.25),(5.25,5.50,5.75),(5.75,6.00,6.25),(6.25,6.50,6.75),(6.75,7.00,7.25),(7.25,7.50,7.75),(7.75,8.00,8.25),(8.25,8.5,8.75),(8.75,9.0,9.25),(9.25,9.5,9.75),(9.75,10.0,10.25),(10.25,10.5,10.75),(10.75,11.0,11.25),(11.25,11.5,11.75),(11.75,12.0,12.25),(12.25,12.5,12.75),(12.75,13.0,13.25),(13.25,13.5,13.75),(13.75,13.875,14.0)))

logZ_mid = [-2.5, -2.05, -1.75, -1.45, -1.15, -0.85, -0.55, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.4, 0.5]


def singleSP(iz, age=1., mtot=1e8, zred=0.01):
    ''' construct spectra of single stellar population with single metallicity, no dust, 
    no nonsense from firefly SSPs (for consistency). metallicity set by z_arr[iz] where z_arr is
    '''
    z_arr = np.array([0.5, 1.0, 2.0, 10**-1.301, 10**-1.302, 10**-2.301, 10**-2.302, 10**-0.6, 
        10**-0.9, 10**-1.2, 10**-1.6, 10**-1.9]) 
    z_strs = np.array(['z001', 'z002', 'z004', 'z0001.bhb', 'z0001.rhb', 'z10m4.bhb', 'z10m4.rhb', 
        'z-0.6', 'z-0.9', 'z-1.2', 'z-1.6', 'z-1.9']) 
    z_metal = z_arr[iz]

    f_ssp = os.environ['STELLARPOPMODELS_DIR']+'/data/SSP_M11_MILES/ssp_M11_MILES.cha'+z_strs[iz]
    model_age, model_wave, model_flux = np.loadtxt(f_ssp, unpack=True, usecols=[0,2,3]) 

    isage = (model_age == age)
    wave = model_wave[isage] # wavelength
    flux = model_flux[isage] # flux [ergs/s/A]
    
    cosmo = FlatLambdaCDM(70., 0.3) 
    wave *= (1. + zred)
    flux *= mtot / (4.*np.pi * cosmo.luminosity_distance(zred).to(U.cm).value**2)

    fig = plt.figure(figsize=(10,4))
    sub = fig.add_subplot(111)
    sub.plot(wave, flux, c='k', lw=1)
    sub.set_xlabel('observed-frame wavelength [$A$]', fontsize=15) 
    sub.set_xlim([3500., 1e4]) 
    sub.set_ylabel('flux [$ergs/s/cm^2/A$]', fontsize=15) 
    sub.set_ylim([0., flux[(wave > 3000.) & (wave < 1e4)].max()])
    fig.savefig(''.join([UT.fig_dir(), 'SSP.iz', str(iz), '.age', str(age), '.z', str(zred), '.png']), bbox_inches='tight') 

    # save spectra 
    f_spec = ''.join([UT.dat_dir(), 'SSP.iz', str(iz), '.age', str(age), '.z', str(zred), '.dat']) 
    np.savetxt(f_spec, np.vstack([wave, flux]).T)
    return None 


def singleSP_myfirefly(iz, age=1., mtot=1e8, zred=0.01):
    ''' fit the single stellar population file from `singleSP` using firefly 
    '''
    z_arr = np.array([0.5, 1.0, 2.0, 10**-1.301, 10**-1.302, 10**-2.301, 10**-2.302, 10**-0.6, 
        10**-0.9, 10**-1.2, 10**-1.6, 10**-1.9]) 

    f_spec = ''.join([UT.dat_dir(), 'SSP.iz', str(iz), '.age', str(age), '.z', str(zred), '.dat']) 
    wave, flux = np.loadtxt(f_spec, unpack=True, usecols=[0,1]) 
    wave_rest = wave/(1.+zred)
    err = flux * 0.001

    cosmo = FlatLambdaCDM(70., 0.3) 
    ffly = Fitters.myFirefly(cosmo, 
            model='m11', 
            model_lib='MILES', 
            imf='cha', 
            dust_corr=False,                        # no dust correction
            age_lim=[0., cosmo.age(zred).value],    # can't have older populations
            logZ_lim=[np.log10(z_arr[iz])-0.1, np.log10(z_arr[iz])+0.1])
    mask = ffly.emissionlineMask(wave_rest)
    ff_fit, ff_prop = ffly.Fit(wave, flux, err, zred, mask=mask, flux_unit=1.)
    
    print('total mass = %f' % ff_prop['logM_total']) 
    for i in range(ff_prop['n_ssp']): 
        print('--- SSP %i; weight=%f ---' % (i, ff_prop['weightLight_ssp_'+str(i)]))
        print('age = %f'% ff_prop['age_ssp_'+str(i)])
        print('log Z = %f'% ff_prop['logZ_ssp_'+str(i)])
        print('log M_tot = %f' % ff_prop['logM_total_ssp_'+str(i)])

    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.plot(ff_fit['wavelength'], ff_fit['flux'], label='SSP') 
    sub.plot(ff_fit['wavelength'], ff_fit['flux_model'], label='myFirefly best-fit') 
    sub.plot(ff_fit['wavelength'], ff_fit['flux'] - ff_fit['flux_model'], label='residual') 
    #sub.plot(wave_rest, flux, c='k', ls=':') 
    sub.set_xlim([3e3, 1e4]) 
    fig.savefig(''.join([UT.fig_dir(), 'myfirefly.SSP.iz', str(iz), '.age', str(age), '.z', str(zred), '.png']), 
            bbox_inches='tight')
    return None 


def singleSP_SFH(iz, zred=0.01):
    ''' construct spectra with single metallicity, with eagle SFH, no dust, no nonsense  
    '''
    z_arr = np.array([0.5, 1.0, 2.0, 10**-1.301, 10**-1.302, 10**-2.301, 10**-2.302, 10**-0.6, 
        10**-0.9, 10**-1.2, 10**-1.6, 10**-1.9]) 
    z_strs = np.array(['z001', 'z002', 'z004', 'z0001.bhb', 'z0001.rhb', 'z10m4.bhb', 'z10m4.rhb', 
        'z-0.6', 'z-0.9', 'z-1.2', 'z-1.6', 'z-1.9']) 
    z_metal = z_arr[iz]

    # read in SFH and M_* 
    f_eagle = h5py.File(''.join([UT.dat_dir(), '0EAGLE_SFRHs.hdf5']), 'r') 
    i_zmid = np.abs(z_arr[iz] - 10**np.array(logZ_mid)).argmin()
    logZ = logZ_mid[i_zmid] 
    t = age_bounds[:,1]
    dt = age_bounds[:,2] - age_bounds[:,0]
    sfh = f_eagle['SFRH'].value[0,:,i_zmid]

    f_ssp = os.environ['STELLARPOPMODELS_DIR']+'/data/SSP_M11_MILES/ssp_M11_MILES.cha'+z_strs[iz]
    model_age, model_wave, model_flux = np.loadtxt(f_ssp, unpack=True, usecols=[0,2,3]) 
    
    age_uniq = np.sort(np.unique(model_age))
    age_bin = np.concatenate([[0], 0.5*(age_uniq[1:] + age_uniq[:-1])])
    
    mtot_bin, flux_bin = [], [] 
    for i_a in range(len(age_bin)-1): 
        in_agebin = ((14.-t >= age_bin[i_a]) & (14-t < age_bin[i_a+1]))
        mtot_i = np.sum(dt[in_agebin] * sfh[in_agebin] * 1e9)
        mtot_bin.append(mtot_i) 

        isage = (model_age == age_uniq[i_a])
        flux_i = model_flux[isage] * mtot_i
        flux_bin.append(flux_i)
        if i_a == 0: 
            wave = model_wave[isage] # wavelength
            flux = flux_i 
        else: 
            flux += flux_i 
    flux_bin = np.array(flux_bin)

    cosmo = FlatLambdaCDM(70., 0.3) 
    wave *= (1. + zred)
    flux /= (4.*np.pi * cosmo.luminosity_distance(zred).to(U.cm).value**2)
    flux_bin /= (4.*np.pi * cosmo.luminosity_distance(zred).to(U.cm).value**2)

    assert np.sum(mtot_bin) == np.sum(sfh * dt * 1e9)
    print('log M_tot = %f' % np.log10(np.sum(mtot_bin)))
    
    fig = plt.figure(figsize=(12,4))
    sub = fig.add_subplot(121)
    sub.bar(age_bin[:-1], mtot_bin, width=age_bin[:-1]-age_bin[1:])
    sub.bar(14.-age_bounds[:,0], sfh * dt * 1e9, width=dt)
    sub.set_xlabel('Lookback Time [Gyr]', fontsize=25) 
    sub.set_xlim([0., 14]) 
    sub.set_ylabel("$M_\mathrm{form}$", fontsize=25)
    sub.set_yscale("log") 
    sub.set_ylim([1e7, 5e9]) 

    sub = fig.add_subplot(122)
    for i in range(flux_bin.shape[0]): 
        sub.plot(wave, flux_bin[i])
    sub.plot(wave, flux, c='k', lw=3) 
    sub.set_xlabel('observed-frame wavelength [$A$]', fontsize=20) 
    sub.set_xlim([3500., 1e4]) 
    sub.set_ylabel('flux [$ergs/s/cm^2/A$]', fontsize=20) 
    fig.savefig(''.join([UT.fig_dir(), 'SSP_SFH.iz', str(iz), '.z', str(zred), '.png']), bbox_inches='tight') 

    # save spectra 
    f_spec = ''.join([UT.dat_dir(), 'SSP_SFH.iz', str(iz), '.z', str(zred), '.dat']) 
    np.savetxt(f_spec, np.vstack([wave, flux]).T)
    return None 


def singleSP_SFH_myfirefly(iz, zred=0.01):
    ''' construct spectra with single metallicity, with eagle SFH, no dust, no nonsense  
    '''
    z_arr = np.array([0.5, 1.0, 2.0, 10**-1.301, 10**-1.302, 10**-2.301, 10**-2.302, 10**-0.6, 
        10**-0.9, 10**-1.2, 10**-1.6, 10**-1.9]) 
    z_strs = np.array(['z001', 'z002', 'z004', 'z0001.bhb', 'z0001.rhb', 'z10m4.bhb', 'z10m4.rhb', 
        'z-0.6', 'z-0.9', 'z-1.2', 'z-1.6', 'z-1.9']) 
    z_metal = z_arr[iz]

    # read in SFH and M_* 
    f_eagle = h5py.File(''.join([UT.dat_dir(), '0EAGLE_SFRHs.hdf5']), 'r') 
    i_zmid = np.abs(z_arr[iz] - 10**np.array(logZ_mid)).argmin()
    logZ = logZ_mid[i_zmid] 
    t = age_bounds[:,1]
    dt = age_bounds[:,2] - age_bounds[:,0]
    sfh = f_eagle['SFRH'].value[0,:,i_zmid]

    f_ssp = os.environ['STELLARPOPMODELS_DIR']+'/data/SSP_M11_MILES/ssp_M11_MILES.cha'+z_strs[iz]
    model_age, model_wave, model_flux = np.loadtxt(f_ssp, unpack=True, usecols=[0,2,3]) 
    
    age_uniq = np.sort(np.unique(model_age))
    age_bin = np.concatenate([[0], 0.5*(age_uniq[1:] + age_uniq[:-1])])
    
    mtot_bin, flux_bin = [], [] 
    for i_a in range(len(age_bin)-1): 
        in_agebin = ((14.-t >= age_bin[i_a]) & (14-t < age_bin[i_a+1]))
        mtot_i = np.sum(dt[in_agebin] * sfh[in_agebin] * 1e9)
        mtot_bin.append(mtot_i) 

    assert np.sum(mtot_bin) == np.sum(sfh * dt * 1e9)
    print('log M_tot = %f' % np.log10(np.sum(mtot_bin)))

    # read in spectra 
    f_spec = ''.join([UT.dat_dir(), 'SSP_SFH.iz', str(iz), '.z', str(zred), '.dat']) 
    wave, flux = np.loadtxt(f_spec, unpack=True, usecols=[0,1]) 
    wave_rest = wave/(1.+zred)
    err = flux * 0.001

    cosmo = FlatLambdaCDM(70., 0.3) 
    ffly = Fitters.myFirefly(cosmo, 
            model='m11', 
            model_lib='MILES', 
            imf='cha', 
            dust_corr=False,                        # no dust correction
            age_lim=[0., cosmo.age(zred).value],    # can't have older populations
            logZ_lim=[np.log10(z_arr[iz])-0.1, np.log10(z_arr[iz])+0.1])
    mask = ffly.emissionlineMask(wave_rest)
    ff_fit, ff_prop = ffly.Fit(wave, flux, err, zred, mask=mask, flux_unit=1.)
    
    print('total mass = %f' % ff_prop['logM_total']) 
    ssp_age_bin0, ssp_age_bin1 = [], [] 
    ssp_mtot = [] 
    for i in range(ff_prop['n_ssp']): 
        print('--- SSP %i; weight=%f ---' % (i, ff_prop['weightLight_ssp_'+str(i)]))
        print('age = %f'% ff_prop['age_ssp_'+str(i)])
        print('log Z = %f'% ff_prop['logZ_ssp_'+str(i)])
        print('log M_tot = %f' % ff_prop['logM_total_ssp_'+str(i)])
        i_age_bin = np.abs(age_uniq - ff_prop['age_ssp_'+str(i)]).argmin()
        ssp_age_bin0.append(age_bin[i_age_bin]) 
        ssp_age_bin1.append(age_bin[i_age_bin+1])
        ssp_mtot.append(10**ff_prop['logM_total_ssp_'+str(i)]) 
    
    fig = plt.figure(figsize=(12,4))
    sub = fig.add_subplot(121)
    sub.bar(age_bin[:-1], mtot_bin, width=age_bin[:-1]-age_bin[1:], alpha=0.75)
    sub.bar(ssp_age_bin0, ssp_mtot, width=np.array(ssp_age_bin0)-np.array(ssp_age_bin1), alpha=0.75)
    sub.set_xlabel('Lookback Time [Gyr]', fontsize=25) 
    sub.set_xlim([0., 14]) 
    sub.set_ylabel("$M_\mathrm{form}$", fontsize=25)
    sub.set_yscale("log") 
    sub.set_ylim([1e7, 5e9]) 

    sub = fig.add_subplot(122)
    sub.plot(ff_fit['wavelength'], ff_fit['flux'], label='SSP') 
    sub.plot(ff_fit['wavelength'], ff_fit['flux_model'], label='myFirefly best-fit') 
    sub.plot(ff_fit['wavelength'], ff_fit['flux'] - ff_fit['flux_model'], label='residual') 
    #sub.plot(wave_rest, flux, c='k', ls=':') 
    sub.set_xlim([3e3, 1e4]) 
    fig.savefig(''.join([UT.fig_dir(), 'myfirefly.SSP_SFH.iz', str(iz), '.z', str(zred), '.png']), 
            bbox_inches='tight')
    return None 


if __name__=='__main__':
    #singleSP(0, age=1., mtot=1e8, zred=0.01)
    #singleSP(1, age=1., mtot=1e8, zred=0.01)
    #singleSP_myfirefly(0, age=1., mtot=1e8, zred=0.01)
    #singleSP_myfirefly(1, age=1., mtot=1e8, zred=0.01)
    for i in range(5): 
        singleSP_SFH(i, zred=0.1)
        singleSP_SFH_myfirefly(i, zred=0.1)
