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
# -- desi --
from desispec.io import read_spectra
# -- feasibgs -- 
from feasibgs import forwardmodel as FM 
# -- fomospec -- 
from fomospec import util as UT 
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
    if lib == 'bc03': 
        f_name = 'gal_spectrum_'+str(galid)+'_BGS_template_BC03_Stelib.fits'
    elif lib == 'fsps': 
        f_name = 'gal_spectrum_'+str(galid)+'_BGS_template_FSPS_uvmiles.fits'
    f_inspec = fits.open(''.join([UT.dat_dir(), 'Lgal/templates/', f_name]))
    specin = f_inspec[1].data

    spec_in = {}
    spec_in['redshift'] = f_inspec[0].header['REDSHIFT']
    spec_in['wave'] = specin['wave']
    spec_in['flux'] = specin['flux_dust_nonoise'] * 1e-4 * 1e7 *1e17 #from W/A/m2 to 10e-17 erg/s/cm2/A
    spec_in['flux_dust_nonoise'] = specin['flux_dust_nonoise'] * 1e-4 * 1e7 *1e17 
    spec_in['flux_nodust_nonoise'] = specin['flux_nodust_nonoise'] * 1e-4 * 1e7 *1e17
    return spec_in 


def lgal_bgsSpec(galid, iobs, lib='bc03', obs_sampling='spacefill', overwrite=False): 
    ''' simulate DESI BGS spectra from testGalIDs LGal 
    '''
    obscond = obs_condition(sampling=obs_sampling) 
    if iobs >= obscond.shape[0]: 
        raise ValueError
    f_bgsspec = ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'BGSsim.gal_spectrum_', str(galid), '_BGS_template_BC03_Stelib.',
        'obscond_', obs_sampling, '.', str(iobs+1), 'of', str(obscond.shape[0]), '.fits']) 

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


# --- some plots --- 
def plot_obs_condition(): 
    obscond = obs_condition(sampling='spacefill') 
    airmass, seeing, exptime, moonfrac, moonalt, moonsep, sunalt, sunsep = obscond.T

    fig = plt.figure(figsize=(20,4))
    sub = fig.add_subplot(141)
    sub.scatter(airmass, seeing, c='k', s=10) 
    sub.set_xlabel('Airmass', fontsize=20)  
    sub.set_xlim([1., 2.]) 
    sub.set_ylabel('Seeing', fontsize=20) 
    sub.set_ylim([0.95, 1.75]) 
    
    sub = fig.add_subplot(142)
    sub.scatter(exptime, moonfrac, c='k', s=10) 
    sub.set_xlabel('Exposure Time', fontsize=20)  
    sub.set_xlim([400., 1200.]) 
    sub.set_ylabel('Moon Fraction', fontsize=20) 
    sub.set_ylim([0.5, 1.]) 
    
    sub = fig.add_subplot(143)
    sub.scatter(moonsep, moonalt, c='k', s=10) 
    sub.set_xlabel('Moon Separation', fontsize=20)  
    sub.set_xlim([40., 120.]) 
    sub.set_ylabel('Moon Altitude', fontsize=20) 
    sub.set_ylim([-90, 90.]) 
    
    sub = fig.add_subplot(144)
    sub.scatter(sunsep, sunalt, c='k', s=10) 
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
        sub.plot(w_sky, skybrights[i,:], lw=1) 
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
    for iobs in range(1,8): 
        for galid in galids: 
            lgal_bgsSpec(galid, iobs, lib='bc03', obs_sampling='spacefill')
    #plot_obs_condition() 
    #plot_obs_SkyBrightness()
    #plot_lgal_bgsSpec()
