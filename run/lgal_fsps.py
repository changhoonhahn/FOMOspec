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
# -- fomospec -- 
from fomospec import util as UT 
from fomospec import fomo as FOMO 
from fomospec import fitters as Fitters
# --- plotting --- 
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


def LGal_FSPS_nodust(galid, imf='chabrier', validate=False): 
    ''' simplest spectra. No dust. No emission lines. No nothing. 

    :params galid: 
        id of galaxy in the GQP spectral challenge test set

    :params iobs: 
        index of observing condition 

    :params imf: (default: chabrier) 
        imf of the SSP in FSPS forward model 

    :params obs_sampling: (default: obs_sampling) 
        sampling strategy of the observing parameters

    :params validate: (default: False)
        if true generate plots that validate the spectra
    '''
    from feasibgs import forwardmodel as FM 
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

    # get redshift
    f_lgal = fits.open(''.join([UT.dat_dir(), 'Lgal/templates/', 
        'gal_spectrum_'+str(galid)+'_BGS_template_FSPS_uvmiles.fits'])) 
    zred = f_lgal[0].header['REDSHIFT'] 

    # disk contribution 
    wave_rest, lum_disk = FOMO.FSPS_nodust(t_lb, sfh_disk, Z_disk, imf=imf) 
    wave, flux_disk = FOMO.zSpectrum(wave_rest, lum_disk, zred, cosmo=Planck13) 

    # bulge contribution 
    _, lum_bulge = FOMO.FSPS_nodust(t_lb, sfh_bulge, Z_bulge, imf=imf) 
    wave, flux_bulge = FOMO.zSpectrum(wave_rest, lum_bulge, zred, cosmo=Planck13) 

    flux_total = np.sum(flux_disk, axis=0) + np.sum(flux_bulge, axis=0)
    
    # write out galaxy source spectrum 
    f_output = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'LGAL.', str(galid), '.FSPS.nodust.imf_', imf, '.hdf5']) 
    fh5 = h5py.File(f_output, 'w') 
    # store metadata 
    fh5.attrs['logM_disk'] = logM_disk
    fh5.attrs['logM_bulge'] = logM_bulge
    fh5.attrs['logM_total'] = logM_total
    fh5.attrs['tage'] = t_lb   # lookback time (age)
    fh5.attrs['sfh_disk'] = sfh_disk 
    fh5.attrs['sfh_bulge'] = sfh_bulge
    fh5.attrs['Z_disk'] = Z_disk 
    fh5.attrs['Z_bulge'] = Z_bulge
    fh5.attrs['zred'] = zred
    fh5.create_dataset('wavelength_rest', data=wave_rest) 
    fh5.create_dataset('wavelength', data=wave)         # observe-frame wavelength
    fh5.create_dataset('flux', data=flux_total)         # flux of spectrum [erg/s/cm^2/A]
    fh5.create_dataset('flux_disk', data=flux_disk)     # flux of disk SSPs 
    fh5.create_dataset('flux_bulge', data=flux_bulge)   # flux of bulge SSPs
    fh5.close() 
    if not validate: return None 

    # validation plot 
    fig = plt.figure(figsize=(12,10))
    gs = mpl.gridspec.GridSpec(2,2, figure=fig) 
    sub = plt.subplot(gs[0,:]) # flux plot
    for m, fi in zip(sfh_bulge, flux_bulge):
        if m > 0.: sub.plot(wave, fi, ls='--', lw=0.1)
    sub.plot(wave, np.sum(flux_bulge, axis=0), c='C0', ls='--', label='Bulge')
    for m, fi in zip(sfh_disk, flux_disk):
        if m > 0.: sub.plot(wave, fi, ls='--', lw=0.1)
        sub.plot(wave, fi, lw=0.1)
    sub.plot(wave, np.sum(flux_disk, axis=0), c='C1', label='Disk')
    sub.plot(wave, flux_total, c='k', ls=':', label='Total')
    sub.set_xlabel('observed-frame wavelength', fontsize=25)
    sub.set_xlim([3e3, 1e4])
    sub.set_ylabel('flux [$erg/s/cm^2/A$]', fontsize=25)
    sub.set_ylim([-1e-18, 1e-15])
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


def BGSspectra_LGal_FSPS_nodust(galid, iobs=0, imf='chabrier', obs_sampling='spacefill', validate=False): 
    '''
    '''
    # read in galaxy source spectrum and metadata
    f_source = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'LGAL.', str(galid), '.FSPS.nodust.imf_', imf, '.hdf5']) 
    fh5 = h5py.File(f_source, 'w') 
    wave = fh5['wavelength'].value 
    flux = fh5['flux_total'].value 
    flux_disk = fh5['flux_disk'].value
    flux_bulge = fh5['flux_bulge'].value
    zred = fh5.attrs['zred']
    logM_disk = fh5.attrs['logM_disk']
    logM_bulge = fh5.attrs['logM_bulge']
    logM_total = fh5.attrs['logM_total']
    zred = fh5.attrs['zred']
    t_lb = fh5.attrs['tage']    # lookback time (age)
    sfh_disk = fh5.attrs['sfh_disk']
    sfh_bulge = fh5.attrs['sfh_bulge']
    Z_disk = fh5.attrs['Z_disk']
    Z_bulge = fh5.attrs['Z_bulge'] 
    fh5.close()

    # BGS -like spectrum
    wmin, wmax = 3523.0, 9923.0
    wave_source = np.arange(wmin, wmax, 0.2)
    flux_source_interp = sp.interpolate.interp1d(wave, flux_total * 1e17, fill_value='extrapolate') 
    flux_source = flux_source_interp(wave_source)
    
    # read in observing conditions
    obscond = obs_condition(sampling=obs_sampling) 
    airmass, seeing, exptime, _, _, _, _, _ = obscond[iobs,:]
    # read in sky surface brightness
    w_sky, skybrights = obs_SkyBrightness(sampling=obs_sampling)
    skybright = np.clip(skybrights[iobs,:], 0, None) * 1e-17 * U.erg / U.angstrom / U.arcsec**2 / U.cm**2 / U.second

    f_bgs = ''.join([f_source.rsplit('.hdf5',1)[0], '.BGS.', obs_sampling, '_obs', str(iobs), '.fits'])
    # simulate BGS spectra 
    fdesi = FM.fakeDESIspec()
    # BGS spectra output file 
    bgs_spec = fdesi.simExposure(
            wave_source, 
            np.atleast_2d(flux_source), 
            exptime=exptime, 
            airmass=airmass, 
            seeing=seeing, 
            skycondition={'name': 'input', 'sky': skybright, 'wave': w_sky}, 
            filename=f_bgs)
    if not validate: return None 

    # validation plot 
    fig = plt.figure(figsize=(12,10))
    gs = mpl.gridspec.GridSpec(2,2, figure=fig) 
    sub = plt.subplot(gs[0,:]) # flux plot
    for band in ['b', 'r', 'z']: 
        sub.plot(bgs_spec.wave[band], 1e-17*bgs_spec.flux[band][0], color='gray', lw=0.25)
    for m, fi in zip(sfh_bulge, flux_bulge):
        if m > 0.: sub.plot(wave, fi, ls='--', lw=0.1)
    sub.plot(wave, np.sum(flux_bulge, axis=0), c='C0', ls='--', label='Bulge')
    for m, fi in zip(sfh_disk, flux_disk):
        if m > 0.: sub.plot(wave, fi, ls='--', lw=0.1)
        sub.plot(wave, fi, lw=0.1)
    sub.plot(wave, np.sum(flux_disk, axis=0), c='C1', label='Disk')
    sub.plot(wave, flux_total, c='k', ls=':', label='Total')
    sub.set_xlabel('observed-frame wavelength', fontsize=25)
    sub.set_xlim([3e3, 1e4])
    sub.set_ylabel('flux [$erg/s/cm^2/A$]', fontsize=25)
    sub.set_ylim([-1e-18, 1e-15])
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
    fig.savefig(f_bgs.rsplit('.hdf5', 1)[0]+'.png', bbox_inches='tight') 
    plt.close() 
    return None 


def mFF_spectra_LGal_FSPS_nodust(galid, imf='chabrier'):
    _fspec = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'LGAL.', str(galid), '.FSPS.nodust.imf_', imf, '.hdf5'])
    f_spec = h5py.File(_fspec, 'r') 
    wave_rest = f_spec['wavelength_rest'].value # rest-frame
    wave = f_spec['wavelength'].value # observed-frame 
    flux = f_spec['flux'].value       # erg/s/cm^2/A
    zred = f_spec.attrs['zred'] 
    
    ffly = Fitters.myFirefly(
            Planck13, # comsology
            model='m11', 
            model_lib='MILES', 
            imf='cha', 
            dust_corr=False,                            # no dust correction
            age_lim=[0., 13.],     # can't have older populations
            logZ_lim=[-3.,1.])
    mask = ffly.emissionlineMask(wave_rest) # mask out emission lines 

    f_mff = ''.join([_fspec.rsplit('/', 1)[0], '/myFF.', _fspec.rsplit('/', 1)[1].rsplit('.hdf5',1)[0], '.source.hdf5']) 
    ff_fit, ff_prop = ffly.Fit(wave, flux, flux*0.001, zred, mask=mask, f_output=f_mff, flux_unit=1.)

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
    fig.savefig(''.join([f_mff.rsplit('.hdf5',1)[0], '.png']), bbox_inches='tight') 
    plt.close() 
    return None


def mFF_BGSspectra_LGal_FSPS_nodust(galid, iobs, imf='chabrier', obs_sampling='spacefill'): 
    ''' generate spectra from LGal_FSPS_nodust
    '''
    # read source spectra for meta data
    f_lgal = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 'LGAL.', str(galid), '.FSPS.nodust.imf_', imf, '.hdf5'])
    f_source = h5py.File(f_lgal, 'r') 
    zred = f_source.attrs['zred'] 
    
    # read bgs spectra 
    f_bgs = ''.join([f_lgal.rsplit('.hdf5',1)[0], '.BGS.', obs_sampling, '_obs', str(iobs), '.fits'])
    bgs_spec = UT.readDESIspec(f_bgs)
    wave_bgs = np.concatenate([bgs_spec['wave_'+b] for b in ['b', 'r', 'z']])      # observed frame
    flux_bgs = np.concatenate([bgs_spec['flux_'+b][0] for b in ['b', 'r', 'z']])   # 10-17 ergs/s/cm2/AA
    flux_err_bgs = np.concatenate([bgs_spec['ivar_'+b][0]**-0.5 for b in ['b', 'r', 'z']])
    
    wavesort = np.argsort(wave_bgs) # sort by wavelenght
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
    f_mff = ''.join([f_bgs.rsplit('/', 1)[0], '/mFF.', f_bgs.rsplit('/', 1)[1].rsplit('.fits', 1)[0], '.hdf5'])
    ff_fit, ff_prop = ffly.Fit(wave_bgs, flux_bgs, flux_err_bgs, zred, mask=mask, f_output=f_mff, flux_unit=1e-17)

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
    fig.savefig(''.join([f_mff.rsplit('.hdf5',1)[0], '.png']), bbox_inches='tight') 
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
    for gid in galids[5:]: 
        #LGal_FSPS_nodust(gid, imf='chabrier', validate=False)
        #mFF_spectra_LGal_FSPS_nodust(gid, imf='chabrier')
        for iobs in [0]: # for different observing conditions 
            BGSspectra_LGal_FSPS_nodust(gid, iobs=iobs, imf='chabrier', obs_sampling='spacefill', validate=True)
            mFF_BGSspectra_LGal_FSPS_nodust(gid, iobs, imf='chabrier', obs_sampling='spacefill')
