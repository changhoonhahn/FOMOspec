'''

code for the spectral challenge of the GQP working group 

'''
import os
import h5py 
import pickle 
import numpy as np 
import scipy as sp 
from astropy.io import fits 
from astropy import units as u
import astropy.cosmology as co
from astropy.table import Table
# -- desi --
from desispec.io import read_spectra
# -- feasibgs -- 
from feasibgs import forwardmodel as FM 
# -- fomospec -- 
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

####################################################
# firefly fitting
####################################################
def firefly_lgal_sourceSpec(galid, lib='bc03', model='m11', model_lib='MILES', imf='cha', hpf_mode='on'): 
    ''' run firefly on simulated source spectra from testGalIDs LGal
    '''
    # read in source spectra
    specin = lgal_sourceSpectra(galid, lib=lib)
    redshift = specin['meta']['REDSHIFT']
    
    gspec = Spec.GSfirefly()
    gspec.generic(specin['wave'], specin['flux_dust_nonoise'],  redshift=redshift)
    gspec.path_to_spectrum = UT.dat_dir()

    # output firefly file 
    f_specin = f_Source(galid, lib=lib)
    f_firefly = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'firefly.', model, '.', model_lib, '.imf_', imf, '.dust_', hpf_mode, '__', 
        f_specin.rsplit('/', 1)[1].rsplit('.fits', 1)[0], '.hdf5']) 
    
    firefly = Fitters.Firefly(gspec,
            f_firefly, # output file 
            co.Planck13, # comsology
            models = model, # model ('m11', 'bc03', 'm09') 
            model_libs = [model_lib], # model library for M11
            imfs = [imf], # IMF used ('ss', 'kr', 'cha')
            hpf_mode = hpf_mode, # uses HPF to dereden the spectrum                       
            age_limits = [0, 15], 
            Z_limits = [-3., 5.], 
            wave_limits = [3350., 9000.], 
            suffix=None, 
            downgrade_models = False, 
            data_wave_medium = 'vacuum', 
            use_downgraded_models = False, 
            write_results = True)
    bestfit = firefly.fit_models_to_data()
    return None 


def firefly_lgal_bgsSpec(galid, iobs, lib='bc03', obs_sampling='spacefill', 
        model='m11', model_lib='MILES', imf='cha', hpf_mode='on'): 
    ''' run firefly on simulated DESI BGS spectra from testGalIDs LGal
    '''
    obscond = obs_condition(sampling=obs_sampling) 
    if iobs >= obscond.shape[0]: raise ValueError

    # read in source spectra
    f_inspec = fits.open(f_Source(galid, lib=lib))
    redshift = f_inspec[0].header['REDSHIFT']
    
    # read in simulated DESI bgs spectra
    f_bgsspec = f_BGSspec(galid, iobs, lib=lib, obs_sampling=obs_sampling, nobs=obscond.shape[0])
    if not os.path.isfile(f_bgsspec): raise ValueError('spectra file does not exist')
    spec_desi = read_spectra(f_bgsspec)
    
    gspec = Spec.GSfirefly()
    gspec.DESIlike(spec_desi, redshift=redshift)
    gspec.path_to_spectrum = UT.dat_dir()

    # output firefly file 
    f_bgs = f_BGSspec(galid, iobs, lib=lib, obs_sampling=obs_sampling, nobs=obscond.shape[0])
    f_firefly = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'firefly.', model, '.', model_lib, '.imf_', imf, '.dust_', hpf_mode, '__', 
        f_bgs.rsplit('/', 1)[1].rsplit('.fits', 1)[0], '.hdf5']) 

    firefly = Fitters.Firefly(gspec,
            f_firefly, # output file 
            co.Planck13, # comsology
            models = model, # model ('m11', 'bc03', 'm09') 
            model_libs = [model_lib], # model library for M11
            imfs = [imf], # IMF used ('ss', 'kr', 'cha')
            hpf_mode = hpf_mode, # uses HPF to dereden the spectrum                       
            age_limits = [0, 15], 
            Z_limits = [-3., 5.], 
            wave_limits = [3350., 9000.], 
            suffix=None, 
            downgrade_models = False, 
            data_wave_medium = 'vacuum', 
            use_downgraded_models = False, 
            write_results = True)
    bestfit = firefly.fit_models_to_data()
    return None 


def firefly_lgal_bgsSpec_validate(galid, iobs, lib='bc03', obs_sampling='spacefill', 
        model='m11', model_lib='MILES', imf='cha', hpf_mode='on'): 
    obscond = obs_condition(sampling=obs_sampling) 
    if iobs >= obscond.shape[0]: raise ValueError
    
    # read in source spectra and meta data 
    specin = lgal_sourceSpectra(galid, lib=lib)
    zred = specin['meta']['REDSHIFT'] 
    
    # read in simulated DESI bgs spectra
    _spec_desi = lgal_bgsSpec(galid, iobs, lib=lib, obs_sampling=obs_sampling) 
    spec_desi = {}  
    spec_desi['wave'] = np.concatenate([_spec_desi.wave[b] for b in ['b', 'r', 'z']])
    spec_desi['flux'] = np.concatenate([_spec_desi.flux[b][0] for b in ['b', 'r', 'z']]) # 10-17 ergs/s/cm2/AA
    spec_desi['flux_unc'] = np.concatenate([_spec_desi.ivar[b][0]**-0.5 for b in ['b', 'r', 'z']])

    # read firefly fitting output  
    f_bgs = f_BGSspec(galid, iobs, lib=lib, obs_sampling=obs_sampling, nobs=obscond.shape[0])
    f_firefly = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'firefly.', model, '.', model_lib, '.imf_', imf, '.dust_', hpf_mode, '__', 
        f_bgs.rsplit('/', 1)[1].rsplit('.fits', 1)[0], '.hdf5']) 
    ffly_out, ffly_prop = UT.readFirefly(f_firefly) 
    
    # spectra comparison
    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(111)
    sub.plot(spec_desi['wave'], spec_desi['flux'], c='k', lw=0.5, label='DESI-like')
    sub.plot(specin['wave'], specin['flux'], c='C0', lw=1, label='Source')
    sub.plot(ffly_out['wavelength'] * (1. + zred), ffly_out['flux_bestfit'], c='C1', label='FIREFLY best-fit')
    sub.legend(loc='upper right', fontsize=20)
    sub.set_xlabel('Wavelength [$\AA$]', fontsize=25)
    sub.set_xlim([spec_desi['wave'].min(), spec_desi['wave'].max()])
    sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/\AA$]', fontsize=25)
    sub.set_ylim([0., 2.*specin['flux'][(specin['wave'] > spec_desi['wave'].min()) & (specin['wave'] < spec_desi['wave'].max())].max()])
    fig.savefig(''.join([f_firefly.rsplit('/', 1)[0], '/', 
        '_spectra.', f_firefly.rsplit('/', 1)[1].rsplit('.', 1)[0], '.png']), bbox_inches='tight') 
    
    # compare inferred properties to input properties
    gal_input = LGalInput(galid) 
    fig = plt.figure(figsize=(7,4))
    sub = fig.add_subplot(111)
    sub.plot(gal_input['sfh_t'], gal_input['sfh_disk'], color='C0', label='disk')
    sub.plot(gal_input['sfh_t'], gal_input['sfh_bulge'], color='red', label='bulge')
    sub.plot(gal_input['sfh_t'], gal_input['sfh_bulge'] + gal_input['sfh_disk'], color='black', label='total')
    mmed = 10**ffly_prop['stellar_mass']
    sub.plot([0., 15.], [mmed, mmed], c='k', ls='--', label='Firefly')
    sub.set_xlabel('Lookback time (Gyr)', fontsize=25)
    sub.set_xlim([1e-2, 13.])
    sub.set_ylabel(r'Mass formed (M$_\odot$)', fontsize=25)
    sub.set_yscale('log')
    sub.set_ylim([1e8, 5e10])
    fig.savefig(''.join([f_firefly.rsplit('/', 1)[0], '/', 
        '_mstar.', f_firefly.rsplit('/', 1)[1].rsplit('.', 1)[0], '.png']), bbox_inches='tight') 
    return None


def firefly_Mstar(lib='bc03', obs_sampling='spacefill', model='m11', model_lib='MILES', imf='cha', hpf_mode='on'):
    ''' see how well Mstar inferred from firefly reproduces the 
    input mstar 
    '''
    obscond = obs_condition(sampling=obs_sampling) 
    galids = testGalIDs()
    obs = range(8)

    mstar_ffly_source, mstar_ffly_obs = [], [] 
    mtot_input, mdisk_input, mbulg_input = [], [], [] 
    for iobs in obs: 
        mstar_ffly = []
        for galid in galids: 
            # read firefly fitting output  
            f_bgs = f_BGSspec(galid, iobs, lib=lib, obs_sampling=obs_sampling, nobs=obscond.shape[0])
            f_firefly = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
                'firefly.', model, '.', model_lib, '.imf_', imf, '.dust_', hpf_mode, '__', 
                f_bgs.rsplit('/', 1)[1].rsplit('.fits', 1)[0], '.hdf5']) 
            ffly_out, ffly_prop = UT.readFirefly(f_firefly) 
            mstar_ffly.append(10**ffly_prop['stellar_mass'])
            
            if iobs == 0: 
                gal_input = LGalInput(galid) 
                mdisk_input.append(np.sum(gal_input['sfh_disk']))
                mbulg_input.append(np.sum(gal_input['sfh_bulge']))
                mtot_input.append(np.sum(gal_input['sfh_disk'])+np.sum(gal_input['sfh_bulge']))

                # source firefly file 
                f_specin = f_Source(galid, lib=lib)
                f_firefly_s = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
                    'firefly.', model, '.', model_lib, '.imf_', imf, '.dust_', hpf_mode, '__', 
                    f_specin.rsplit('/', 1)[1].rsplit('.fits', 1)[0], '.hdf5']) 
                ffly_out, ffly_prop = UT.readFirefly(f_firefly_s) 
                mstar_ffly_source.append(10**ffly_prop['stellar_mass'])

        mstar_ffly_obs.append(mstar_ffly) 
    
    fig = plt.figure(figsize=(7,7))
    sub = fig.add_subplot(111)
    sub.hist(np.log10(mtot_input), range=(9,12), bins=20, histtype='step', color='k', linewidth=2)  
    sub.hist(np.log10(mstar_ffly_source), range=(9,12), bins=20, histtype='step', 
            color='k', linewidth=1, linestyle=':')
    for iobs in obs: 
        sub.hist(np.log10(mstar_ffly_obs[iobs]), range=(9,12), bins=20, histtype='step', 
                color='C'+str(iobs), linewidth=1)
    sub.set_xlabel(r'$\log(\,M_*$ [$M_\odot$]\,)', fontsize=25) 
    sub.set_xlim([9, 12])
    fig.savefig(''.join([UT.fig_dir(), 
        'mstar_hist.', f_firefly.rsplit('/', 1)[1].rsplit('__', 1)[0], '.png']), bbox_inches='tight') 

    fig = plt.figure(figsize=(7,7))
    sub = fig.add_subplot(111)
    sub.scatter(mtot_input, mstar_ffly_source, s=5, color='C'+str(iobs))
    for iobs in obs: 
        sub.scatter(mtot_input, mstar_ffly_obs[iobs], s=5, color='C'+str(iobs))
    sub.plot([1e8, 1e12], [1e8,1e12], c='k', ls='--')
    sub.set_xlabel(r'$M_*^\mathrm{(input)}$ [$M_\odot$]', fontsize=25) 
    sub.set_xscale('log')
    sub.set_xlim([1e8, 1e12])
    sub.set_ylabel(r'$M_*^\mathrm{(firefly)}$ [$M_\odot$]', fontsize=25) 
    sub.set_yscale('log')
    sub.set_ylim([1e8, 1e12])
    fig.savefig(''.join([UT.fig_dir(), 
        'mstar_comparison.', f_firefly.rsplit('/', 1)[1].rsplit('__', 1)[0], '.png']), bbox_inches='tight') 

    fig = plt.figure(figsize=(7,7))
    sub = fig.add_subplot(111)
    for iobs in obs: 
        sub.hist(np.log10(mtot_input) - np.log10(mstar_ffly_obs[iobs]), range=(0,5), bins=20, histtype='step', 
                color='C'+str(iobs), linewidth=1)
    sub.set_xlabel(r'$\log(\,M_*^\mathrm{(input)}\,)-\,\log(\,M_*^\mathrm{(firefly)}\,)$ ', fontsize=25) 
    sub.set_xlim([9, 12])
    fig.savefig(''.join([UT.fig_dir(), 
        'dmstar_hist.', f_firefly.rsplit('/', 1)[1].rsplit('__', 1)[0], '.png']), bbox_inches='tight') 
    return None 

####################################################
# forward modeling DESI BGS-like test spectra
####################################################
def f_Source(galid, lib='bc03'): 
    ''' source spectra
    '''
    if lib == 'bc03': lib_str = 'BC03_Stelib'
    elif lib == 'fsps': lib_str = 'FSPS_uvmiles'
    return ''.join([UT.dat_dir(), 'Lgal/templates/', 
        'gal_spectrum_'+str(galid)+'_BGS_template_', lib_str, '.fits'])


def f_BGSspec(galid, iobs, lib='bc03', obs_sampling='spacefill', nobs=8):
    ''' DESI BGS-like spectra
    '''
    fsource = f_Source(galid, lib=lib)
    fsource = fsource.rsplit('/',1)[1].rsplit('.', 1)[0] 
    return ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'BGSsim.', fsource, '.obscond_', obs_sampling, '.', str(iobs+1), 'of', str(nobs), '.fits']) 

def LGalInput(galid): 
    f_input = ''.join([UT.dat_dir(), 'Lgal/gal_inputs/', 
        'gal_input_' + str(galid) + '_BGS_template_FSPS_uvmiles.csv']) 
    gal_input = Table.read(f_input, delimiter=' ')
    return gal_input


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
            specsim_sky.moon.moon_zenith = (90. - moonalt) * u.deg
            specsim_sky.moon.separation_angle = moonsep * u.deg
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


def lgal_sourceSpectra(galid, lib='bc03'): 
    ''' source spectra of LGal object from Rita
    '''
    # read in source spectra
    f_inspec = fits.open(f_Source(galid, lib=lib))
    hdr = f_inspec[0].header
    specin = f_inspec[1].data
    meta = {}
    for k in hdr.keys(): 
        meta[k] = hdr[k]

    spec_in = {}
    spec_in['redshift'] = f_inspec[0].header['REDSHIFT']
    spec_in['wave'] = specin['wave']
    spec_in['flux'] = specin['flux_dust_nonoise'] * 1e-4 * 1e7 *1e17 #from W/A/m2 to 10e-17 erg/s/cm2/A
    spec_in['flux_dust_nonoise'] = specin['flux_dust_nonoise'] * 1e-4 * 1e7 *1e17 
    spec_in['flux_nodust_nonoise'] = specin['flux_nodust_nonoise'] * 1e-4 * 1e7 *1e17
    spec_in['meta'] = meta
    return spec_in 


def lgal_bgsSpec(galid, iobs, lib='bc03', obs_sampling='spacefill', overwrite=False): 
    ''' simulate DESI BGS spectra from testGalIDs LGal 
    '''
    obscond = obs_condition(sampling=obs_sampling) 
    if iobs >= obscond.shape[0]: 
        raise ValueError
    f_bgsspec = f_BGSspec(galid, iobs, lib=lib, obs_sampling=obs_sampling, nobs=obscond.shape[0])

    if os.path.isfile(f_bgsspec) and not overwrite: 
        bgs_spectra = read_spectra(f_bgsspec) 
    else: 
        # get source spectra 
        spec_source = lgal_sourceSpectra(galid, lib=lib)
        wavemin, wavemax = 3523.0, 9923.0
        wave_source = np.arange(wavemin, wavemax, 0.2)
        flux_source_interp = sp.interpolate.interp1d(
                spec_source['wave'], spec_source['flux'], fill_value='extrapolate') 
        flux_source = flux_source_interp(wave_source)
        
        # observing condition 
        airmass, seeing, exptime, _, _, _, _, _ = obscond[iobs,:]
        # read in sky surface brightness
        w_sky, skybrights = obs_SkyBrightness(sampling=obs_sampling)
        skybright = skybrights[iobs,:]

        u_surface_brightness = 1e-17 * u.erg / u.angstrom / u.arcsec**2 / u.cm**2 / u.second

        # simulate BGS spectra 
        fdesi = FM.fakeDESIspec()
        # BGS spectra output file 
        bgs_spectra = fdesi.simExposure(
                wave_source, 
                np.atleast_2d(flux_source), 
                exptime=exptime, 
                airmass=airmass, 
                seeing=seeing, 
                skycondition={'name': 'input', 
                    'sky': np.clip(skybright, 0, None) * u_surface_brightness, 
                    'wave': w_sky}, 
                filename=f_bgsspec)
    return bgs_spectra 


def plot_obs_condition(): 
    obscond = obs_condition(sampling='spacefill') 
    airmass, seeing, exptime, moonfrac, moonalt, moonsep, sunalt, sunsep = obscond.T

    fig = plt.figure(figsize=(20,4))
    sub = fig.add_subplot(141)
    for i in range(len(airmass)): 
        sub.scatter([airmass[i]], [seeing[i]], c='C'+str(i), s=20) 
    sub.set_xlabel('Airmass', fontsize=20)  
    sub.set_xlim([1., 2.]) 
    sub.set_ylabel('Seeing', fontsize=20) 
    sub.set_ylim([0.95, 1.75]) 
    
    sub = fig.add_subplot(142)
    for i in range(len(airmass)): 
        sub.scatter([exptime[i]], [moonfrac[i]], c='C'+str(i), s=20) 
    sub.set_xlabel('Exposure Time', fontsize=20)  
    sub.set_xlim([400., 1200.]) 
    sub.set_ylabel('Moon Fraction', fontsize=20) 
    sub.set_ylim([0.5, 1.]) 
    
    sub = fig.add_subplot(143)
    for i in range(len(airmass)): 
        sub.scatter([moonsep[i]], [moonalt[i]], c='C'+str(i), s=20) 
    sub.set_xlabel('Moon Separation', fontsize=20)  
    sub.set_xlim([40., 120.]) 
    sub.set_ylabel('Moon Altitude', fontsize=20) 
    sub.set_ylim([-90, 90.]) 
    
    sub = fig.add_subplot(144)
    for i in range(len(airmass)): 
        sub.scatter([sunsep[i]], [sunalt[i]], c='C'+str(i), s=10) 
    sub.set_xlabel('Sun Separation', fontsize=20)  
    sub.set_xlim([40., 180.]) 
    sub.set_ylabel('Sun Altitude', fontsize=20) 
    sub.set_ylim([-90, 0.]) 
    
    fig.subplots_adjust(wspace=0.5)
    fig.savefig(''.join([UT.fig_dir(), 'spectral_challenge.obs_condition.png']), 
            bbox_inches='tight') 
    return None 


def plot_obs_SkyBrightness():
    w_sky, skybrights = obs_SkyBrightness(sampling='spacefill')
    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(111)
    for i in range(skybrights.shape[0]):
        sub.plot(w_sky, skybrights[i,:], lw=1, c='C'+str(i)) 
    sub.set_xlabel(r'Wavelenght [$\AA$]', fontsize=25) 
    sub.set_xlim([3600., 9800.]) 
    sub.set_ylabel(r'Sky Surface Brightness', fontsize=25) 
    sub.set_ylim([0., 20.]) 

    fig.savefig(''.join([UT.fig_dir(), 'spectral_challenge.obs_SkyBrightness.png']), 
            bbox_inches='tight') 
    return None


def plot_lgal_bgsSpec(): 
    galids = testGalIDs()

    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(111)
    for i, galid in enumerate(galids[:10]): 
        spec_source = lgal_sourceSpectra(galid, lib='bc03')
        spec_i = lgal_bgsSpec(galid, 0, lib='bc03', obs_sampling='spacefill')
        for band in ['b', 'r', 'z']: 
            sub.plot(spec_i.wave[band], spec_i.flux[band][0], lw=0.25, c='C'+str(i))
        sub.plot(spec_source['wave'], spec_source['flux'], c='k', ls='--', lw=0.5)
    sub.set_xlabel(r'Wavelenght [$\AA$]', fontsize=25) 
    sub.set_xlim([3600., 9800.]) 
    sub.set_ylabel(r'Flux [$10^{-17} ergs/s/cm^2/\AA$]', fontsize=25) 
    sub.set_ylim([0., 10.]) 
    fig.savefig(''.join([UT.fig_dir(), 'spectral_challenge.lgal_source_bgsSpec.png']), 
            bbox_inches='tight') 

    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(111)
    for i, galid in enumerate(galids[:10]): 
        spec_i = lgal_bgsSpec(galid, 0, lib='bc03', obs_sampling='spacefill')
        for band in ['b', 'r', 'z']: 
            sub.plot(spec_i.wave[band], spec_i.flux[band][0], lw=0.25, c='C'+str(i))
    sub.set_xlabel(r'Wavelenght [$\AA$]', fontsize=25) 
    sub.set_xlim([3600., 9800.]) 
    sub.set_ylabel(r'Flux [$10^{-17} ergs/s/cm^2/\AA$]', fontsize=25) 
    sub.set_ylim([0., 10.]) 
    fig.savefig(''.join([UT.fig_dir(), 'spectral_challenge.lgal_bgsSpec.png']), 
            bbox_inches='tight') 

    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(111)
    snrs = [] 
    for iobs in range(8): 
        spec_i = lgal_bgsSpec(galids[0], iobs, lib='bc03', obs_sampling='spacefill')
        band = 'b' 
        these = ((spec_i.wave[band] > np.mean(spec_i.wave[band])-50) &
                (spec_i.wave[band] < np.mean(spec_i.wave[band])+50) &
                (spec_i.flux[band][0] > 0))
        snr = np.median(spec_i.flux[band][0, these] * np.sqrt(spec_i.ivar[band][0, these]))
        snrs.append(snr) 
    i_sort = np.argsort(snrs) 
    for iobs in i_sort: 
        spec_i = lgal_bgsSpec(galids[0], iobs, lib='bc03', obs_sampling='spacefill')
        for band in ['b', 'r', 'z']: 
            sub.plot(spec_i.wave[band], spec_i.flux[band][0], lw=1, c='C'+str(iobs))
    sub.set_xlabel(r'Wavelenght [$\AA$]', fontsize=25) 
    sub.set_xlim([3600., 9800.]) 
    sub.set_ylabel(r'Flux [$10^{-17} ergs/s/cm^2/\AA$]', fontsize=25) 
    sub.set_ylim([0., 10.]) 

    fig.savefig(''.join([UT.fig_dir(), 'spectral_challenge.lgal_bgsSpec.iobs.png']), 
            bbox_inches='tight') 

    return None


if __name__=="__main__": 
    #obs_condition(sampling='spacefill', overwrite=True)
    #obs_SkyBrightness(sampling='spacefill', overwrite=True)
    galids = testGalIDs()
    for iobs in [0]: #range(1,8): 
        for galid in galids: 
            #lgal_bgsSpec(galid, iobs, lib='bc03', obs_sampling='spacefill')
            firefly_lgal_sourceSpec(galid, lib='bc03', hpf_mode='on')
            #firefly_lgal_bgsSpec(galid, iobs, lib='bc03', obs_sampling='spacefill', hpf_mode='on')
            #firefly_lgal_bgsSpec_validate(galid, iobs, lib='bc03', obs_sampling='spacefill', hpf_mode='on')
    firefly_Mstar() 
    #plot_obs_condition() 
    #plot_obs_SkyBrightness()
    #plot_lgal_bgsSpec()
