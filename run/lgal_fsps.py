'''

generate galaxy spectra from SF and Z histories of LGal with FSPS.


'''
import os 
import h5py
import fsps
import pickle 
import numpy as np 
import scipy as sp 
from astropy.io import fits 
from astropy import units as U 
from astropy.cosmology import Planck13 
# -- desi --
from desispec.io import read_spectra
# -- feasibgs -- 
from feasibgs import forwardmodel as FM 
# -- fomospec -- 
from fomospec import util as UT 
from fomospec import fitters as Fitters
# --- plotting --- 
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
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


def LGal_FSPS_nodust(galid, imf='chabrier', validate=False): 
    ''' simplest spectra. No dust.  
    '''
    i_imf = {'chabrier': 1} 
    # get Lgal galaxy SF and Z histories 
    f_input = ''.join([UT.dat_dir(), 'Lgal/gal_inputs/', 'gal_input_', str(galid), '_BGS_template_FSPS_uvmiles.csv'])
    lgal = np.loadtxt(f_input, skiprows=1, unpack=True, delimiter=' ')

    t_lb        = lgal[0] # lookback time [Gyr] 
    sfh_disk    = lgal[2] # M* formed in disk at t_lb 
    Z_disk      = lgal[4] # Z of disk at t_lb
    sfh_bulge   = lgal[3] # M* formed in bulge at t_lb
    Z_bulge     = lgal[5]  # Z of bulge at t_lb 

    logM_disk = np.log10(np.sum(sfh_disk))
    logM_bulge = np.log10(np.sum(sfh_bulge))
    logM_total = np.log10(np.sum(sfh_disk) + np.sum(sfh_bulge))

    print('disk: log(Mformed) = %f' % logM_disk)
    print('bulge: log(Mformed) = %f' % logM_bulge)
    print('total: log(Mformed) = %f' % logM_total)

    flux_disk, flux_bulge = [], [] 
    # disk contribution 
    hasM_disk = (sfh_disk > 0.) 
    t_lb_disk = t_lb[hasM_disk]
    for t, m, z in zip(t_lb_disk, sfh_disk[hasM_disk], Z_disk[hasM_disk]): 
        ssp_i = fsps.StellarPopulation(
                zcontinuous=1,          # interpolate metallicities
                sfh=0,                  # returns SSP
                logzsol=np.log10(z),    # log(Z) 
                imf_type=i_imf[imf])    # default is chabrier 

        wave_rest, flux_i = ssp_i.get_spectrum(tage=t, peraa=True) # in units of Lsun/AA
        flux_disk.append(m * flux_i)    # Mformed * SSP 
    flux_disk = np.array(flux_disk) 

    # bulge contribution 
    hasM_bulge = (sfh_bulge > 0.) 
    t_lb_bulge = t_lb[hasM_bulge]
    for t, m, z in zip(t_lb_bulge, sfh_bulge[hasM_bulge], Z_bulge[hasM_bulge]): 
        ssp_i = fsps.StellarPopulation(
                zcontinuous=1,          # interpolate metallicities
                sfh=0,                  # returns SSP
                logzsol=np.log10(z),    # log(Z) 
                imf_type=1)             # chabrier IMF for now 
        
        wave_rest, flux_i = ssp_i.get_spectrum(tage=t, peraa=True) # in units of Lsun/AA
        flux_bulge.append(m * flux_i)    # Mformed * SSP 
    flux_bulge = np.array(flux_bulge) 

    # output luminosity  
    f_output = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'LGAL.', str(galid), '.FSPS.nodust.imf_', imf, '.hdf5']) 
    fh5 = h5py.File(f_output, 'w') 
    fh5.create_dataset('wavelength', data=wave_rest)    # rest-frame wavelength [AA]
    fh5.create_dataset('sfh_disk', data=sfh_disk[hasM_disk]) 
    fh5.create_dataset('flux_disk', data=flux_disk)     # disk SSP fluxes [Lsun/AA]
    fh5.create_dataset('flux_bulge', data=flux_bulge)   # bulge SSP fluxes [Lsun/AA]
    # store metadata 
    fh5.attrs['logM_disk'] = logM_disk
    fh5.attrs['logM_bulge'] = logM_bulge
    fh5.attrs['logM_total'] = logM_total
    fh5.attrs['tage'] = t_lb   # lookback time (age)
    fh5.attrs['sfh_disk'] = sfh_disk 
    fh5.attrs['sfh_bulge'] = sfh_bulge
    fh5.attrs['Z_disk'] = Z_disk 
    fh5.attrs['Z_bulge'] = Z_bulge
    fh5.close() 
    if not validate: return None 

    # validation plot 
    fig = plt.figure(figsize=(12,10))
    gs = mpl.gridspec.GridSpec(2,2, figure=fig) 
    sub = plt.subplot(gs[0,:]) # flux plot
    if np.sum(hasM_bulge) > 0: 
        for fi in flux_bulge:
            sub.plot(wave_rest, fi, ls='--', lw=0.1)
        sub.plot(wave_rest, np.sum(flux_bulge, axis=0), c='C0', ls='--', label='Bulge')
    if np.sum(hasM_disk) > 0: 
        for fi in flux_disk:
            sub.plot(wave_rest, fi, lw=0.1)
        sub.plot(wave_rest, np.sum(flux_disk, axis=0), c='C1', label='Disk')
    sub.plot(wave_rest, np.sum(flux_bulge, axis=0) + np.sum(flux_disk, axis=0), c='k', ls=':', label='Total')
    sub.set_xlabel('rest-frame wavelength', fontsize=25)
    sub.set_xlim([3e3, 1e4])
    sub.set_ylabel('flux [$L_\odot/A$]', fontsize=25)
    sub.set_ylim([-1e6, 1e7])
    sub.legend(loc='upper right', fontsize=20)
    
    sub = plt.subplot(gs[1,0]) # SFH
    sub.plot(t_lb, sfh_bulge, c='C0', label='Bulge')
    sub.plot(t_lb, sfh_disk, c='C1', label='Disk')
    sub.text(0.05, 0.95, ('$M_\mathrm{tot}=10^{%.1f}, M_\mathrm{disk}=10^{%.1f}, M_\mathrm{bulge}=10^{%.1f}$' % 
        (logM_total, logM_disk, logM_bulge)), ha='left', va='top', transform=sub.transAxes, fontsize=15)
    sub.set_xlabel('Lookback Time [$Gyr$]', fontsize=20)
    sub.set_xlim([0., 13.])
    sub.set_ylabel("$M_\mathrm{formed}$", fontsize=25)
    sub.set_yscale("log")
    sub.set_ylim([5e6, 5e10]) 
    
    sub = plt.subplot(gs[1,1]) # ZH
    sub.plot(t_lb, Z_bulge, c='C0', label='Bulge')
    sub.plot(t_lb, Z_disk, c='C1', label='Disk')
    sub.set_xlabel('Lookback Time [$Gyr$]', fontsize=20)
    sub.set_xlim([0., 13.])
    sub.set_ylabel("metallicity, $Z$", fontsize=25)
    sub.set_yscale('log') 
    sub.set_ylim([1e-3, 1e-1]) 
    fig.subplots_adjust(wspace=0.25, hspace=0.25)
    fig.savefig(f_output.rsplit('.hdf5', 1)[0]+'.png', bbox_inches='tight') 
    plt.close() 
    return None 


def spectra_LGal_FSPS_nodust(galid, imf='chabrier', validate=False): 
    ''' generate spectra from LGal_FSPS_nodust
    '''
    f_lgal = fits.open(''.join([UT.dat_dir(), 'Lgal/templates/', 
        'gal_spectrum_'+str(galid)+'_BGS_template_FSPS_uvmiles.fits'])) 
    zred = f_lgal[0].header['REDSHIFT'] # redshift 

    # fsps 
    f_fsps = h5py.File(''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'LGAL.', str(galid), '.FSPS.nodust.imf_', imf, '.hdf5']), 'r') 

    wave = f_fsps['wavelength'].value * (1. + zred) # redshift wavelength 
    if f_fsps['flux_disk'].value.size != 0: 
        flux_disk = f_fsps['flux_disk'].value * UT.Lsun() / (4. * np.pi * Planck13.luminosity_distance(zred).to(U.cm).value**2) 
    else: 
        flux_disk = np.zeros((1, len(wave)))
    if f_fsps['flux_bulge'].value.size != 0: 
        flux_bulge = f_fsps['flux_bulge'].value * UT.Lsun() / (4. * np.pi * Planck13.luminosity_distance(zred).to(U.cm).value**2) 
    else: 
        flux_bulge = np.zeros((1, len(wave)))
    flux_total = np.sum(flux_disk, axis=0) + np.sum(flux_bulge, axis=0)
    
    # write out galaxy spectrum 
    f_output = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'LGAL.', str(galid), '.FSPS.nodust.imf_', imf, '.spectrum.source.hdf5']) 
    fh5 = h5py.File(f_output, 'w') 
    fh5.attrs['zred'] = zred
    for k in f_fsps.attrs.keys(): 
        fh5.attrs[k] = f_fsps.attrs[k] 
    fh5.create_dataset('wavelength', data=wave)         # observe-frame wavelength
    fh5.create_dataset('flux', data=flux_total)         # flux of spectrum [erg/s/cm^2/A]
    fh5.create_dataset('flux_disk', data=flux_disk)     # flux of disk SSPs 
    fh5.create_dataset('flux_bulge', data=flux_bulge)   # flux of bulge SSPs
    fh5.close() 
    if not validate: return None 

    # validation plot 
    fig = plt.figure(figsize=(12,10))
    sub = fig.add_subplot(111)
    for fi in flux_bulge:
        sub.plot(wave, fi, ls='--', lw=0.1)
    sub.plot(wave, np.sum(flux_bulge, axis=0), c='C0', ls='--', label='Bulge')
    for fi in flux_disk:
        sub.plot(wave, fi, lw=0.1)
    sub.plot(wave, np.sum(flux_disk, axis=0), c='C1', label='Disk')
    sub.plot(wave, flux_total, c='k', ls=':', label='Total')
    sub.set_xlabel('observed-frame wavelength', fontsize=25)
    sub.set_xlim([3e3, 1e4])
    sub.set_ylabel('flux [$erg/s/cm^2/A$]', fontsize=25)
    sub.set_ylim([-1e-18, 1e-15])
    sub.legend(loc='upper right', fontsize=20)
    fig.savefig(f_output.rsplit('.hdf5',1)[0]+'.png', bbox_inches='tight') 
    plt.close() 
    return None 


def BGSspectra_LGal_FSPS_nodust(galid, iobs, imf='chabrier', obs_sampling='spacefill', validate=False): 
    ''' generate spectra from LGal_FSPS_nodust
    '''
    # source galaxy spectrum 
    f_source = h5py.File(''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'LGAL.', str(galid), '.FSPS.nodust.imf_', imf, '.spectrum.source.hdf5']), 'r') 

    wavemin, wavemax = 3523.0, 9923.0
    wave_source = np.arange(wavemin, wavemax, 0.2)
    flux_source_interp = sp.interpolate.interp1d(
            f_source['wavelength'].value, 1e17*f_source['flux'].value, fill_value='extrapolate') 
    flux_source = flux_source_interp(wave_source)
    
    # observing condition 
    obscond = obs_condition(sampling=obs_sampling) 
    assert iobs < obscond.shape[0] 
    airmass, seeing, exptime, _, _, _, _, _ = obscond[iobs,:]
    # read in sky surface brightness
    w_sky, skybrights = obs_SkyBrightness(sampling=obs_sampling)
    skybright = skybrights[iobs,:]

    u_sb = 1e-17 * U.erg / U.angstrom / U.arcsec**2 / U.cm**2 / U.second

    f_bgs = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'LGAL.', str(galid), '.FSPS.nodust.imf_', imf, '.spectrum.BGS.', obs_sampling, '_obs', str(iobs), '.fits'])
    # simulate BGS spectra 
    fdesi = FM.fakeDESIspec()
    # BGS spectra output file 
    bgs_spec = fdesi.simExposure(
            wave_source, 
            np.atleast_2d(flux_source), 
            exptime=exptime, 
            airmass=airmass, 
            seeing=seeing, 
            skycondition={'name': 'input', 
                'sky': np.clip(skybright, 0, None) * u_sb, 
                'wave': w_sky}, 
            filename=f_bgs)
    if not validate: return None 

    # validation plot 
    fig = plt.figure(figsize=(12,6))
    sub = fig.add_subplot(111)
    for band in ['b', 'r', 'z']: 
        sub.plot(bgs_spec.wave[band], bgs_spec.flux[band][0], lw=0.25)
    sub.plot(wave_source, flux_source, c='k', ls='--', lw=0.5)
    sub.set_xlabel('observed-frame wavelength', fontsize=25)
    sub.set_xlim([wavemin, wavemax])
    sub.set_ylabel('flux [$10^{-17}\,erg/s/cm^2/A$]', fontsize=25)
    sub.set_ylim([0., None]) 
    sub.legend(loc='upper right', fontsize=20)
    fig.savefig(f_bgs.rsplit('.fits',1)[0]+'.png', bbox_inches='tight') 
    plt.close() 
    return None 


def mFF_spectra_LGal_FSPS_nodust(galid, imf='chabrier'):
    f_spec = h5py.File(''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'LGAL.', str(galid), '.FSPS.nodust.imf_', imf, '.spectrum.source.hdf5']), 'r') 
    wave = f_spec['wavelength'].value # observed-frame 
    flux = f_spec['flux'].value       # erg/s/cm^2/A
    zred = f_spec.attrs['zred'] 
    wave_rest = wave / (1.+zred) 
    
    ffly = Fitters.myFirefly(
            Planck13, # comsology
            model='m11', 
            model_lib='MILES', 
            imf='cha', 
            dust_corr=False,                            # no dust correction
            age_lim=[0., 13.],     # can't have older populations
            logZ_lim=[-3.,1.])
    mask = ffly.emissionlineMask(wave_rest)
    ff_fit, ff_prop = ffly.Fit(wave, flux, flux*0.001, zred, mask=mask, flux_unit=1.)

    print('input: total_mass = %f' % f_spec.attrs['logM_total'])  
    print('firefly: total mass = %f' % ff_prop['logM_total']) 
    _ssp_age, _ssp_z, _ssp_mtot = [], [], [] 
    for i in range(ff_prop['n_ssp']): 
        print('--- SSP %i; weight=%f ---' % (i, ff_prop['weightLight_ssp_'+str(i)]))
        print('age = %f'% ff_prop['age_ssp_'+str(i)])
        print('log Z = %f'% ff_prop['logZ_ssp_'+str(i)])
        print('log M_tot = %f' % ff_prop['logM_total_ssp_'+str(i)])
        _ssp_age.append(ff_prop['age_ssp_'+str(i)]) 
        _ssp_z.append(10**ff_prop['logZ_ssp_'+str(i)]) 
        _ssp_mtot.append(10**ff_prop['logM_total_ssp_'+str(i)]) 
    agesort = np.argsort(_ssp_age)
    ssp_age = np.array(_ssp_age)[agesort]
    ssp_z = np.array(_ssp_z)[agesort]
    ssp_mtot = np.array(_ssp_mtot)[agesort]

    # validation plot 
    fig = plt.figure(figsize=(12,10))
    gs = mpl.gridspec.GridSpec(2,2, figure=fig) 
    sub = plt.subplot(gs[0,:]) # flux plot
    sub.plot(wave, flux, c='C0', lw=1, label='LGal spectrum')
    sub.plot(ff_fit['wavelength'] * (1. + zred), ff_fit['flux_model'], c='C1', label='FIREFLY best-fit')
    sub.plot(ff_fit['wavelength'] * (1. + zred), ff_fit['flux'] - ff_fit['flux_model'], c='r', label='residual') 
    sub.text(0.05, 0.95, ('$M_\mathrm{tot}=10^{%.2f}; M_\mathrm{firefly} = 10^{%.2f}$' % 
        (f_spec.attrs['logM_total'], ff_prop['logM_total'])), ha='left', va='top', transform=sub.transAxes, fontsize=20)
    sub.set_xlabel('observed-frame wavelength', fontsize=25)
    sub.set_xlim([3e3, 1e4])
    sub.set_ylabel('flux [$erg/s/cm^2/A$]', fontsize=25)
    sub.legend(loc='upper right', fontsize=20)
    
    sub = plt.subplot(gs[1,0]) # SFH
    sub.plot(f_spec.attrs['tage'], f_spec.attrs['sfh_bulge'], c='C0', ls=':', label='Bulge')
    sub.plot(f_spec.attrs['tage'], f_spec.attrs['sfh_disk'], c='C0', ls='--', label='Disk')
    sub.plot(f_spec.attrs['tage'], f_spec.attrs['sfh_bulge'] + f_spec.attrs['sfh_disk'], c='C0', label='Total')
    sub.plot(ssp_age, ssp_mtot, c='C1')
    sub.set_xlabel('Lookback Time [$Gyr$]', fontsize=20)
    sub.set_xlim([0., 13.])
    sub.set_ylabel("$M_\mathrm{formed}$", fontsize=25)
    sub.set_yscale("log")
    sub.set_ylim([5e6, 5e10]) 
    
    sub = plt.subplot(gs[1,1]) # ZH
    sub.plot(f_spec.attrs['tage'], f_spec.attrs['Z_bulge'], c='C0', ls=':', label='Bulge')
    sub.plot(f_spec.attrs['tage'], f_spec.attrs['Z_disk'], c='C0', ls='--', label='Disk')
    sub.plot(ssp_age, ssp_z, c='C1')
    sub.set_xlabel('Lookback Time [$Gyr$]', fontsize=20)
    sub.set_xlim([0., 13.])
    sub.set_ylabel("metallicity, $Z$", fontsize=25)
    sub.set_yscale('log') 
    sub.set_ylim([1e-3, 1e-1]) 
    fig.subplots_adjust(wspace=0.25, hspace=0.25)
    fig.savefig(''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'mFF.LGAL.', str(galid), '.FSPS.nodust.imf_', imf, '.spectrum.source.png']), bbox_inches='tight') 
    plt.close() 
    return None


def mFF_BGSspectra_LGal_FSPS_nodust(galid, iobs, imf='chabrier', obs_sampling='spacefill'): 
    ''' generate spectra from LGal_FSPS_nodust
    '''
    # read source spectra for meta data
    f_source = h5py.File(''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'LGAL.', str(galid), '.FSPS.nodust.imf_', imf, '.spectrum.source.hdf5']), 'r') 
    zred = f_source.attrs['zred'] 
    
    # read bgs spectra 
    f_bgs = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'LGAL.', str(galid), '.FSPS.nodust.imf_', imf, '.spectrum.BGS.', obs_sampling, '_obs', str(iobs), '.fits'])
    bgs_spec = read_spectra(f_bgs) 
    wave_bgs = np.concatenate([bgs_spec.wave[b] for b in ['b', 'r', 'z']])      # observed frame
    flux_bgs = np.concatenate([bgs_spec.flux[b][0] for b in ['b', 'r', 'z']])   # 10-17 ergs/s/cm2/AA
    flux_err_bgs = np.concatenate([bgs_spec.ivar[b][0]**-0.5 for b in ['b', 'r', 'z']])

    wavesort = np.argsort(wave_bgs)
    wave_bgs = wave_bgs[wavesort]
    flux_bgs = flux_bgs[wavesort]
    flux_err_bgs = flux_err_bgs[wavesort]

    ffly = Fitters.myFirefly(
            Planck13, # comsology
            model='m11', 
            model_lib='MILES', 
            imf='cha', 
            dust_corr=False,                            # no dust correction
            age_lim=[0., 13.],     # can't have older populations
            logZ_lim=[-3.,1.])
    mask = ffly.emissionlineMask(wave_bgs/(1.+zred))
    ff_fit, ff_prop = ffly.Fit(wave_bgs, flux_bgs, flux_err_bgs, zred, mask=mask, flux_unit=1e-17)

    print('input: total_mass = %f' % f_source.attrs['logM_total'])  
    print('firefly: total mass = %f' % ff_prop['logM_total']) 
    _ssp_age, _ssp_z, _ssp_mtot = [], [], [] 
    for i in range(ff_prop['n_ssp']): 
        print('--- SSP %i; weight=%f ---' % (i, ff_prop['weightLight_ssp_'+str(i)]))
        print('age = %f'% ff_prop['age_ssp_'+str(i)])
        print('log Z = %f'% ff_prop['logZ_ssp_'+str(i)])
        print('log M_tot = %f' % ff_prop['logM_total_ssp_'+str(i)])
        _ssp_age.append(ff_prop['age_ssp_'+str(i)]) 
        _ssp_z.append(10**ff_prop['logZ_ssp_'+str(i)]) 
        _ssp_mtot.append(10**ff_prop['logM_total_ssp_'+str(i)]) 
    agesort = np.argsort(_ssp_age)
    ssp_age = np.array(_ssp_age)[agesort]
    ssp_z = np.array(_ssp_z)[agesort]
    ssp_mtot = np.array(_ssp_mtot)[agesort]

    # validation plot 
    fig = plt.figure(figsize=(12,10))
    gs = mpl.gridspec.GridSpec(2,2, figure=fig) 
    sub = plt.subplot(gs[0,:]) # flux plot
    sub.plot(wave_bgs, flux_bgs, c='C0', lw=1, label='LGal BGS spectrum')
    sub.plot(ff_fit['wavelength'] * (1. + zred), ff_fit['flux_model'], c='C1', label='FIREFLY best-fit')
    sub.plot(ff_fit['wavelength'] * (1. + zred), ff_fit['flux'] - ff_fit['flux_model'], c='r', label='residual') 
    sub.text(0.05, 0.95, ('$M_\mathrm{tot}=10^{%.2f}; M_\mathrm{firefly} = 10^{%.2f}$' % 
        (f_source.attrs['logM_total'], ff_prop['logM_total'])), ha='left', va='top', transform=sub.transAxes, fontsize=20)
    sub.set_xlabel('observed-frame wavelength', fontsize=25)
    sub.set_xlim([3e3, 1e4])
    sub.set_ylabel('flux [$erg/s/cm^2/A$]', fontsize=25)
    sub.legend(loc='upper right', fontsize=20)
    
    sub = plt.subplot(gs[1,0]) # SFH
    sub.plot(f_source.attrs['tage'], f_source.attrs['sfh_bulge'], c='C0', ls=':', label='Bulge')
    sub.plot(f_source.attrs['tage'], f_source.attrs['sfh_disk'], c='C0', ls='--', label='Disk')
    sub.plot(f_source.attrs['tage'], f_source.attrs['sfh_bulge'] + f_source.attrs['sfh_disk'], c='C0', label='Total')
    sub.plot(ssp_age, ssp_mtot, c='C1')
    sub.set_xlabel('Lookback Time [$Gyr$]', fontsize=20)
    sub.set_xlim([0., 13.])
    sub.set_ylabel("$M_\mathrm{formed}$", fontsize=25)
    sub.set_yscale("log")
    sub.set_ylim([5e6, 5e10]) 
    
    sub = plt.subplot(gs[1,1]) # ZH
    sub.plot(f_source.attrs['tage'], f_source.attrs['Z_bulge'], c='C0', ls=':', label='Bulge')
    sub.plot(f_source.attrs['tage'], f_source.attrs['Z_disk'], c='C0', ls='--', label='Disk')
    sub.plot(ssp_age, ssp_z, c='C1')
    sub.set_xlabel('Lookback Time [$Gyr$]', fontsize=20)
    sub.set_xlim([0., 13.])
    sub.set_ylabel("metallicity, $Z$", fontsize=25)
    sub.set_yscale('log') 
    sub.set_ylim([1e-3, 1e-1]) 
    fig.subplots_adjust(wspace=0.25, hspace=0.25)
    fig.savefig(''.join([f_bgs.rsplit('/', 1)[0], '/', 'mFF.', f_bgs.rsplit('/', 1)[1].rsplit('.fits', 1)[0], '.png']),
            bbox_inches='tight') 
    plt.close() 
    return None 


def testGalIDs(): 
    ''' get gal IDs for test set of LGal SAM objects
    '''
    # read in spectral challenge test set filenames set by Rita 
    f_test = ''.join([UT.dat_dir(), 'spectral_challenge/', 'lgal_filenames_testset_BC03_Stellib.txt']) 
    fnames_test = np.loadtxt(f_test, unpack=True, dtype='S', skiprows=1)  
    # get gal ID's from the filename 
    galid_test = [int(fname.split('_')[2]) for fname in fnames_test]
    return galid_test


def obs_condition(sampling='spacefill', overwrite=False): 
    ''' Sample mock BGS observing conditions 
    '''
    if sampling == 'spacefill': 
        f_obscond = ''.join([UT.dat_dir(), 'spectral_challenge/', 
            'bgs_mockexp_obscond_testset.', sampling, '.txt'])
    else: 
        raise NotImplementedError("maybe implement random later") 

    if os.path.isfile(f_obscond) and not overwrite: 
        out_obscond = np.loadtxt(f_obscond, unpack=True, skiprows=1)  
    else:
        # read in bgs mock exposure observing conditions generated from 
        # https://github.com/changhoonhahn/feasiBGS/blob/master/notebook/local_bgs_mockexp.ipynb 
        f_exp = ''.join([UT.dat_dir(), 'spectral_challenge/', 
            'bgs_survey_exposures.withsun.hdf5'])
        f = h5py.File(f_exp, 'r') 
        mock_exps = {} 
        for k in f.keys(): 
            mock_exps[k] = f[k].value 
        f.close()
        nexps = len(mock_exps['RA'])
        
        # we want to sample main meta-data from the mock exposures 
        meta_keys = ['MOONFRAC', 'MOONALT', 'SUNALT'] #'AIRMASS', 'SUNSEP', 'MOONSEP' 
        meta = np.zeros((nexps, len(meta_keys)))
        for i, k in enumerate(meta_keys): 
            meta[:,i] = mock_exps[k]

        if sampling == 'spacefill': 
            histmd, edges = np.histogramdd(meta, 2)
            _hasexp = histmd > 0.
            has_exp = np.where(_hasexp)
            print('%i exposures' % np.sum(_hasexp))

            iexp_sample = [] 
            for i in range(np.sum(_hasexp)): 
                in_bin = np.ones(nexps).astype(bool)
                for i_dim in range(len(meta_keys)): 
                    in_bin = (in_bin & 
                        (meta[:,i_dim] > edges[i_dim][has_exp[i_dim]][i]) & 
                        (meta[:,i_dim] <= edges[i_dim][has_exp[i_dim]+1][i]))
                assert np.sum(in_bin) > 0
                iexp_sample.append(np.random.choice(np.arange(nexps)[in_bin], 1)[0])
            iexp_sample = np.array(iexp_sample)

        out_keys = ['AIRMASS', 'SEEING', 'EXPTIME', 'MOONFRAC', 'MOONALT', 'MOONSEP', 'SUNALT', 'SUNSEP'] 
        hdr = ', '.join([k.lower() for k in out_keys])
        out_obscond = np.zeros((len(iexp_sample), len(out_keys)))
        for i, k in enumerate(out_keys): 
            out_obscond[:,i] = mock_exps[k][iexp_sample]
        np.savetxt(f_obscond, out_obscond, header=hdr) 
    return out_obscond.T


def obs_SkyBrightness(sampling='spacefill', overwrite=False): 
    ''' sky brightness of the sampled observing condition 
    '''
    # observing condition 
    obscond = obs_condition(sampling=sampling) 
    
    f_skybright = ''.join([UT.dat_dir(), 'spectral_challenge/', 
        'sky_brightness.bgs_mockexp_obscond_testset.', sampling, '.p'])

    if os.path.isfile(f_skybright) and not overwrite:
        wave, skybrights = pickle.load(open(f_skybright, 'rb'))
        return wave, skybrights 
    else:
        from feasibgs import skymodel as Sky
        specsim_sky = Sky.specsim_initialize('desi')
        specsim_wave = specsim_sky._wavelength # Ang
        
        for i in range(obscond.shape[0]):  
            airmass, seeing, exptime, moonfrac, moonalt, moonsep, sunalt, sunsep = obscond[i,:]
            # re-scaled KS contribution 
            specsim_sky.airmass = airmass
            specsim_sky.moon.moon_phase = np.arccos(2.*moonfrac - 1)/np.pi
            specsim_sky.moon.moon_zenith = (90. - moonalt) * U.deg
            specsim_sky.moon.separation_angle = moonsep * U.deg
            # re-calibrated KS coefficients 
            specsim_sky.moon.KS_CR = 458173.535128
            specsim_sky.moon.KS_CM0 = 5.540103
            specsim_sky.moon.KS_CM1 = 178.141045
            I_ks_rescale = specsim_sky.surface_brightness

            if sunalt > -20.: 
                import pandas as pd
                # there is twilight contributions 
                # read in coefficients from Parker
                fmoon = ''.join([UT.dat_dir(), 'spectral_challenge/', 'MoonResults.csv'])
                coeffs = pd.DataFrame.from_csv(fmoon)
                coeffs.columns = [
                        'wl', 'model', 'data_var', 'unexplained_var',' X2', 'rX2',
                        'c0', 'c_am', 'tau', 'tau2', 'c_zodi', 'c_isl', 'sol', 'I',
                        't0', 't1', 't2', 't3', 't4', 'm0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6',
                        'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                        'c2', 'c3', 'c4', 'c5', 'c6']
                # keep moon models
                twi_coeffs = coeffs[coeffs['model'] == 'twilight']
                coeffs = coeffs[coeffs['model'] == 'moon']
                # order based on wavelengths for convenience
                wave_sort = np.argsort(np.array(coeffs['wl']))

                for k in coeffs.keys():
                    coeffs[k] = np.array(coeffs[k])[wave_sort]

                for k in twi_coeffs.keys():
                    twi_coeffs[k] = np.array(twi_coeffs[k])[wave_sort]

                I_twi = (
                        twi_coeffs['t0'] * np.abs(sunalt) + # CT2
                        twi_coeffs['t1'] * np.abs(sunalt)**2 +      # CT1
                        twi_coeffs['t2'] * np.abs(sunsep)**2 +      # CT3
                        twi_coeffs['t3'] * np.abs(sunsep)           # CT4
                        ) * np.exp(-twi_coeffs['t4'] * airmass) + twi_coeffs['c0']
                I_twi = np.array(I_twi)/np.pi 
                I_twi_interp = sp.interpolate.interp1d(10.*np.array(coeffs['wl']), I_twi, 
                        fill_value='extrapolate')
                skybright = I_ks_rescale.value + I_twi_interp(specsim_wave.value)
            else: 
                skybright = I_ks_rescale.value

            if i == 0:  
                skybrights = np.zeros((obscond.shape[0], len(skybright)))
            skybrights[i,:] = skybright 
        pickle.dump([specsim_wave.value, skybrights], open(f_skybright, 'wb'))
        return specsim_wave.value, skybrights 


if __name__=='__main__': 
    galids = testGalIDs() 
    for gid in galids[:5]: 
        #LGal_FSPS_nodust(gid, imf='chabrier', validate=False)
        #spectra_LGal_FSPS_nodust(gid, imf='chabrier', validate=False)
        #mFF_spectra_LGal_FSPS_nodust(gid, imf='chabrier')
        #BGSspectra_LGal_FSPS_nodust(gid, 0, imf='chabrier', obs_sampling='spacefill', validate=True)
        mFF_BGSspectra_LGal_FSPS_nodust(gid, 0, imf='chabrier', obs_sampling='spacefill')
