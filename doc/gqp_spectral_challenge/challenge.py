'''

plots for the GQP spectral challenge


'''
import os 
import h5py 
import numpy as np 
import scipy as sp 
# -- astropy -- 
from astropy.io import fits 
from astropy import units as u
# -- desi --
from desispec.io import read_spectra
# -- feasibgs -- 
from feasibgs import forwardmodel as FM 
# -- fomospec -- 
from fomospec import util as UT 
from fomospec import fomo as FOMO 
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


def mFF_LGal_nodust(imf='chabrier', iobs=0, obs_sampling='spacefill'): 
    ''' Compares properties inferred from myFirefly to input Lgal properties. 
    The properties we compare are Mformed, mass weighted age, and mass weighted
    metallicity. 

    :param imf: (default: chabrier) 
        imf used to generate spectra for Lgal 
    
    :param iobs: (default: 0) 
        index of BGS observing conditions sampled for the spectral challenge

    :param obs_sampling: (default: 'spacefill') 
        sampling method for the observing conditions. 
    '''
    f_source = lambda gid: ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'LGAL.', str(gid), '.FSPS.nodust.imf_', imf, '.hdf5']) 
    # source spectrum firefly file 
    mff_source = lambda gid: ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'myFF.LGAL.', str(gid), '.FSPS.nodust.imf_', imf, '.source.hdf5'])
    # bgs-like spectrum firefly file 
    mff_bgs = lambda gid: ''.join([UT.dat_dir(), 'spectral_challenge/bgs/mFF.LGAL.', str(gid), '.FSPS.nodust.imf_', imf, 
        '.BGS.', obs_sampling, '_obs', str(iobs), '.hdf5'])

    # gather inferred and input properties
    galids = np.unique(testGalIDs())
    mform_input, age_input, z_input = np.zeros(len(galids)), np.zeros(len(galids)), np.zeros(len(galids)) # input 
    mform_inf_non, age_inf_non, z_inf_non= np.zeros((len(galids), 3)), np.zeros((len(galids), 3)), np.zeros((len(galids), 3)) # no noise
    mform_inf_bgs, age_inf_bgs, z_inf_bgs= np.zeros((len(galids), 3)), np.zeros((len(galids), 3)), np.zeros((len(galids), 3)) # bgs noise

    for i, galid in enumerate(galids): 
        # read in input M_total 
        lgal = h5py.File(f_source(galid), 'r') 
        mform_input[i] = 10**lgal.attrs['logM_total']
        # mass weighted age and metallicity 
        age_input[i] = np.average(lgal.attrs['tage'], weights=lgal.attrs['sfh_disk']+lgal.attrs['sfh_bulge']) 
        z_input[i] = np.average(np.concatenate([lgal.attrs['Z_disk'], lgal.attrs['Z_bulge']]), 
                weights=np.concatenate([lgal.attrs['sfh_disk'], lgal.attrs['sfh_bulge']]))

        # read in source spectrum FF file 
        ffly_out, ffly_prop = UT.readmyFirefly(mff_source(galid)) 
        mform_inf_non[i,1] = 10**ffly_prop['logM_total']
        mform_inf_non[i,0] = 10**ffly_prop['logM_total_up_1sig']
        mform_inf_non[i,2] = 10**ffly_prop['logM_total_low_1sig']
        
        age_inf_non[i,1] = ffly_prop['age_massW']
        age_inf_non[i,0] = ffly_prop['age_massW_up_1sig'] 
        age_inf_non[i,2] = ffly_prop['age_massW_low_1sig']
        z_inf_non[i,1] = 10**ffly_prop['logZ_massW'] * 0.0190
        z_inf_non[i,0] = 10**ffly_prop['logZ_massW_up_1sig'] * 0.0190
        z_inf_non[i,2] = 10**ffly_prop['logZ_massW_lwo_1sig'] * 0.0190
        
        # read in BGS spectrum FF file 
        ffly_out, ffly_prop = UT.readmyFirefly(mff_bgs(galid)) 
        mform_inf_bgs[i,1] = 10**ffly_prop['logM_total']
        mform_inf_bgs[i,0] = 10**ffly_prop['logM_total_up_1sig']
        mform_inf_bgs[i,2] = 10**ffly_prop['logM_total_low_1sig']
        
        age_inf_bgs[i,1] = ffly_prop['age_massW']
        age_inf_bgs[i,0] = ffly_prop['age_massW_up_1sig'] 
        age_inf_bgs[i,2] = ffly_prop['age_massW_low_1sig']
        z_inf_bgs[i,1] = 10**ffly_prop['logZ_massW'] * 0.0190
        z_inf_bgs[i,0] = 10**ffly_prop['logZ_massW_up_1sig'] * 0.0190
        z_inf_bgs[i,2] = 10**ffly_prop['logZ_massW_lwo_1sig'] * 0.0190

    fig = plt.figure(figsize=(20,6))
    # scatter plot of log Mtotal (inferred) vs log Mtotal (input)
    sub = fig.add_subplot(131)
    sub.errorbar(mform_input, mform_inf_non[:,1], 
            yerr=[mform_inf_non[:,1]-mform_inf_non[:,0], mform_inf_non[:,2]-mform_inf_non[:,1]], 
            fmt='.k', markersize=10, elinewidth=2, label='source (no noise)')
    sub.errorbar(mform_input, mform_inf_bgs[:,1], 
            yerr=[mform_inf_bgs[:,1]-mform_inf_bgs[:,0], mform_inf_bgs[:,2]-mform_inf_bgs[:,1]], 
            fmt='.C1', markersize=5, elinewidth=1, label=r'bgs; $i_\mathrm{obs}='+str(iobs+1)+'$')
    sub.plot([1e8, 1e12], [1e8,1e12], c='k', ls='--')
    sub.set_xlabel(r'$M_\mathrm{total}^\mathrm{(input)}$ [$M_\odot$]', fontsize=25) 
    sub.set_xscale('log')
    sub.set_xlim([1e9, 1e12])
    sub.set_ylabel(r'$M_\mathrm{total}^\mathrm{(firefly)}$ [$M_\odot$]', fontsize=25) 
    sub.set_yscale('log')
    sub.set_ylim([1e9, 1e12])
    sub.legend(loc='upper left', markerscale=5, handletextpad=0, fontsize=20) 

    # scatter plot of mass weighted age (inferred) vs (input)
    sub = fig.add_subplot(132)
    sub.errorbar(age_input, age_inf_non[:,1], yerr=[age_inf_non[:,1]-age_inf_non[:,0], age_inf_non[:,2]-age_inf_non[:,1]], 
            fmt='.k', markersize=10, elinewidth=2)
    sub.errorbar(age_input, age_inf_bgs[:,1], yerr=[age_inf_bgs[:,1]-age_inf_bgs[:,0], age_inf_bgs[:,2]-age_inf_bgs[:,1]], 
            fmt='.C1', markersize=5, elinewidth=1)
    sub.plot([0.,15.], [0.,15.], c='k', ls='--')
    sub.set_xlabel(r'mass weighted input age [Gyr]', fontsize=25) 
    sub.set_xlim([0., 12])
    sub.set_ylabel(r'mass weighted firefly age [Gyr]', fontsize=25) 
    sub.set_ylim([0., 12])
    
    # scatter plot of mass weighted Z (inferred) vs (input)
    sub = fig.add_subplot(133)
    sub.errorbar(z_input, z_inf_non[:,1], yerr=[z_inf_non[:,1]-z_inf_non[:,0], z_inf_non[:,2]-z_inf_non[:,1]], 
            fmt='.k', markersize=10, elinewidth=2)
    sub.errorbar(z_input, z_inf_bgs[:,1], yerr=[z_inf_bgs[:,1]-z_inf_bgs[:,0], z_inf_bgs[:,2]-z_inf_bgs[:,1]], 
            fmt='.C1', markersize=5, elinewidth=1)
    sub.plot([0.,1.], [0.,1.], c='k', ls='--')
    sub.set_xlabel(r'mass weighted input Z', fontsize=25) 
    sub.set_xscale('log')
    sub.set_xlim([1e-3, 1e-1])
    sub.set_ylabel(r'mass weighted firefly Z', fontsize=25) 
    sub.set_yscale('log')
    sub.set_ylim([1e-3, 1e-1])
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(''.join([UT.fig_dir(), 'mFF_LGal_nodust.obs', str(iobs), '.png']), bbox_inches='tight') 
    raise ValueError

    fig = plt.figure(figsize=(7,7))
    sub = fig.add_subplot(111)
    sub.hist(np.log10(m_source), range=(9,12), bins=20, histtype='step', color='k', linewidth=2, label='input')  
    sub.hist(np.log10(minf_source), range=(9,12), bins=20, histtype='step', 
            color='k', linewidth=1, linestyle=':', label='Firefly (noiseless)')
    sub.hist(np.log10(minf_bgs), range=(9,12), bins=20, histtype='step', 
            color='r', linewidth=1, linestyle=':', label='Firefly (bgs)')
    sub.set_xlabel(r'$\log(\,M_*$ [$M_\odot$]\,)', fontsize=25) 
    sub.set_xlim([9, 12])
    sub.legend(loc='upper right', frameon=True, fontsize=20) 
    fig.savefig(''.join([UT.fig_dir(), 'mFF_LGal_nodust_Mstar_hist.obs', str(iobs), '.png']), bbox_inches='tight') 

    fig = plt.figure(figsize=(7,7))
    sub = fig.add_subplot(111)
    sub.hist(np.log10(m_source) - np.log10(minf_source), range=(-1.5,1.5), bins=20, color='k', histtype='stepfilled', alpha=0.5, label='noiseless')
    sub.hist(np.log10(m_source) - np.log10(minf_bgs), range=(-1.5,1.5), bins=20, color='r', histtype='stepfilled', alpha=0.5, label='bgs')
    sub.set_xlabel(r'$\log(\,M_*^\mathrm{(input)}\,)-\,\log(\,M_*^\mathrm{(firefly)}\,)$ ', fontsize=25) 
    sub.set_xlim([-1.5, 1.5])
    sub.legend(loc='upper right', fontsize=20) 
    fig.savefig(''.join([UT.fig_dir(), 'mFF_LGal_nodust_dMstar_hist.obs', str(iobs), '.png']), bbox_inches='tight') 
    return None 


def f_nonoise(galid, lib='bc03'): 
    ''' retrun noiseless spectra template file name 
    
    :param galid: 
        galaxy id number

    :param lib: (default: 'bc03') 
        specify stellar population synthesis library. Options are
        'bc03' and 'fsps'
    '''
    if lib == 'bc03': lib_str = 'BC03_Stelib'
    elif lib == 'fsps': lib_str = 'FSPS_uvmiles'
    return os.path.join(UT.dat_dir(), 'Lgal', 'templates', 'gal_spectrum_'+str(galid)+'_BGS_template_'+lib_str+'.fits')


def Lgal_nonoiseSpectra(galid, lib='bc03'): 
    ''' read in noiseless spectra of LGal object generated by Rita
    
    :param galid: 
        galaxy id number

    :param lib: (default: 'bc03') 
        specify stellar population synthesis library. Options are
        'bc03' and 'fsps'

    :return spec_in: 
        dictionary with redshift, relevant meta data and flux
    '''
    # read in source spectra
    f_inspec = fits.open(f_nonoise(galid, lib=lib))
    hdr = f_inspec[0].header
    specin = f_inspec[1].data
    meta = {}
    for k in hdr.keys(): 
        meta[k] = hdr[k]

    spec_in = {}
    spec_in['redshift'] = f_inspec[0].header['REDSHIFT']
    spec_in['wave'] = specin['wave']
    spec_in['flux_dust_nonoise'] = specin['flux_dust_nonoise'] * 1e-4 * 1e7 *1e17 #from W/A/m2 to 10e-17 erg/s/cm2/A
    spec_in['flux_nodust_nonoise'] = specin['flux_nodust_nonoise'] * 1e-4 * 1e7 *1e17
    spec_in['meta'] = meta
    return spec_in 


def f_BGS(galid, iobs, lib='bc03', obs_sampling='spacefill', dust=False):
    ''' file name of DESI BGS-like spectra
    
    :param galid: 
        galaxy id number

    :param iobs: 
        index of sampled observing conditions 

    :param lib: (default: 'bc03') 
        specify stellar population synthesis library. Options are
        'bc03' and 'fsps'

    :param obs_sampling: (default: 'spacefill') 
        method for sampling the observation 

    :param dust: (default: False) 
        whether or not to use the spectra with dust or without 

    :return fbgs: 
        return file name of the BGS 
    '''
    fsource = f_nonoise(galid, lib=lib)
    fsource = fsource.rsplit('/',1)[1].rsplit('.', 1)[0] 
    if dust: str_dust = 'dust'
    else: str_dust = 'nodust'
    fbgs = 'BGSsim.%s.%s.obscond_%s.obs%i.fits' % (fsource, str_dust, obs_sampling, (iobs+1))
    return os.path.join(UT.dat_dir(), 'spectral_challenge', 'bgs', fbgs) 


def Lgal_BGSnoiseSpec(galid, iobs, lib='bc03', obs_sampling='spacefill', dust=False, overwrite=False, validate=False): 
    ''' return (and generate) spectra of LGal object with DESI BGS-like noise based on  
    observing conditions described by iobs and obs_sampling. 
    
    :param galid: 
        galaxy id number

    :param iobs: 
        index of sampled observing conditions 

    :param lib: (default: 'bc03') 
        specify stellar population synthesis library. Options are
        'bc03' and 'fsps'

    :param obs_sampling: (default: 'spacefill') 
        method for sampling the observation 

    :return : 
        dictionary with 
    '''
    # check that iobs is within the sampling range 
    obscond = obs_condition(sampling=obs_sampling) 
    if iobs >= obscond.shape[0]: raise ValueError
    fbgs = f_BGS(galid, iobs, lib=lib, obs_sampling=obs_sampling, dust=dust)

    if os.path.isfile(fbgs) and not overwrite: 
        bgs_spectra = read_spectra(fbgs) 
    else: 
        # read in source spectra 
        spec_source = Lgal_nonoiseSpectra(galid, lib=lib)
        if dust: 
            flux = spec_source['flux_dust_nonoise'] 
        else: 
            flux = spec_source['flux_nodust_nonoise'] 
        wavemin, wavemax = 3523.0, 9923.0
        wave_source = np.arange(wavemin, wavemax, 0.2)
        flux_source_interp = sp.interpolate.interp1d(
                spec_source['wave'], flux, fill_value='extrapolate') 
        flux_source = flux_source_interp(wave_source)
        
        # observing condition that go into the forward model
        airmass, seeing, exptime, _, _, _, _, _ = obscond[iobs,:]
        # read in sky surface brightness
        w_sky, skybrights = obs_SkyBrightness(sampling=obs_sampling)
        skybright = np.clip(skybrights[iobs,:], 0, None) * 1e-17 * u.erg / u.angstrom / u.arcsec**2 / u.cm**2 / u.second

        # simulate BGS spectra 
        fdesi = FM.fakeDESIspec()
        # BGS spectra output file 
        bgs_spectra = fdesi.simExposure(
                wave_source, 
                np.atleast_2d(flux_source), 
                exptime=exptime, 
                airmass=airmass, 
                seeing=seeing, 
                skycondition={'name': 'input', 'sky': skybright, 'wave': w_sky}, 
                filename=fbgs)

    if validate: # generate some plots to validate the spectra  
    
        fig = plt.figure(figsize=(15,5))
        sub = fig.add_subplot(111)
        spec_source = Lgal_nonoiseSpectra(galid, lib=lib)
        for band in ['b', 'r', 'z']: 
            sub.plot(bgs_spectra.wave[band], bgs_spectra.flux[band][0], lw=0.25, c='C1')
        if dust: 
            sub.plot(spec_source['wave'], spec_source['flux_dust_nonoise'], c='k', ls='--', lw=0.5)
        else:
            sub.plot(spec_source['wave'], spec_source['flux_nodust_nonoise'], c='k', ls='--', lw=0.5)
        sub.set_xlabel(r'Wavelenght [$\AA$]', fontsize=25) 
        sub.set_xlim([3600., 9800.]) 
        sub.set_ylabel(r'Flux [$10^{-17} ergs/s/cm^2/\AA$]', fontsize=25) 
        sub.set_ylim([0., 10.]) 
        fig.savefig(fbgs.rsplit('.fits', 1)[0]+'.png', bbox_inches='tight') 
    return bgs_spectra 


def testGalIDs(): 
    ''' get gal IDs for test set of LGal SAM objects
    '''
    # read in spectral challenge test set filenames set by Rita 
    f_test = ''.join([UT.dat_dir(), 'spectral_challenge/', 'lgal_filenames_testset_BC03_Stellib.txt']) 
    fnames_test = np.loadtxt(f_test, unpack=True, dtype='S', skiprows=1)  
    # get gal ID's from the filename 
    galid_test = [int(fname.split('_')[2]) for fname in fnames_test]
    return galid_test


def obs_condition(sampling='spacefill', validate=False): 
    ''' Sample mock BGS observing conditions from `surveysim` output. 

    :param sampling: (default: 'spacefill') 
        method for samplign the `surveysim` observing conditions. The default
        is a hacky spacefilling method  
    '''
    if sampling not in ['spacefill']: 
        raise NotImplementedError("maybe implement random later") 
    
    f_obscond = os.path.join(UT.dat_dir(), 'spectral_challenge', 'bgs_mockexp_obscond_testset.'+sampling+'.txt')

    if os.path.isfile(f_obscond): 
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

    if validate: # some plots to validate the runs 
        airmass, seeing, exptime, moonfrac, moonalt, moonsep, sunalt, sunsep = out_obscond

        fig = plt.figure(figsize=(20,4))
        sub = fig.add_subplot(141)
        for i in range(len(airmass)): 
            sub.scatter([airmass[i]], [seeing[i]], c='k', s=20) 
            sub.text(1.02*airmass[i], 1.02*seeing[i], str(i+1), 
                    ha='left', va='bottom', fontsize=15, color='k', bbox=dict(facecolor='w', alpha=0.7))
        sub.set_xlabel('Airmass', fontsize=20)  
        sub.set_xlim([1., 2.]) 
        sub.set_ylabel('Seeing', fontsize=20) 
        sub.set_ylim([0.95, 1.75]) 
        
        sub = fig.add_subplot(142)
        for i in range(len(airmass)): 
            sub.scatter([exptime[i]], [moonfrac[i]], c='k', s=20) 
            sub.text(1.02*exptime[i], 1.02*moonfrac[i], str(i+1), 
                    ha='left', va='bottom', fontsize=15, color='k', bbox=dict(facecolor='w', alpha=0.7))
        sub.set_xlabel('Exposure Time', fontsize=20)  
        sub.set_xlim([400., 1200.]) 
        sub.set_ylabel('Moon Fraction', fontsize=20) 
        sub.set_ylim([0.5, 1.]) 
        
        sub = fig.add_subplot(143)
        for i in range(len(airmass)): 
            sub.scatter([moonsep[i]], [moonalt[i]], c='k', s=20) 
            sub.text(1.02*moonsep[i], 1.02*moonalt[i], str(i+1), 
                    ha='left', va='bottom', fontsize=15, color='k', bbox=dict(facecolor='w', alpha=0.7))
        sub.set_xlabel('Moon Separation', fontsize=20)  
        sub.set_xlim([40., 120.]) 
        sub.set_ylabel('Moon Altitude', fontsize=20) 
        sub.set_ylim([-90, 90.]) 
        
        sub = fig.add_subplot(144)
        for i in range(len(airmass)): 
            sub.scatter([sunsep[i]], [sunalt[i]], c='k', s=20) 
            sub.text(1.02*sunsep[i], 1.02*sunalt[i], str(i+1), 
                    ha='left', va='bottom', fontsize=15, color='k', bbox=dict(facecolor='w', alpha=0.7))
        sub.set_xlabel('Sun Separation', fontsize=20)  
        sub.set_xlim([40., 180.]) 
        sub.set_ylabel('Sun Altitude', fontsize=20) 
        sub.set_ylim([-90, 0.]) 
        fig.subplots_adjust(wspace=0.5)
        fig.savefig(f_obscond.rsplit('.txt', 1)[0]+'.png', bbox_inches='tight') 

    return out_obscond.T


def obs_SkyBrightness(sampling='spacefill', validate=False): 
    ''' sky brightness of the sampled observing condition generated using 
    obs_condition() above. 

    :param sampling: (Default: 'spacefill') 
        method for sampling the survey sim
    '''
    # observing condition 
    obscond = obs_condition(sampling=sampling) 
    
    f_skybright = os.path.join(UT.dat_dir(), 'spectral_challenge', 
            'sky_brightness.bgs_mockexp_obscond_testset.'+sampling+'.hdf5')

    if os.path.isfile(f_skybright):
        fsb = h5py.File(f_skybright, 'r')
        wave = fsb['wave'].value
        skybrights = fsb['skybrightness'].value
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

            if i == 0: skybrights = np.zeros((obscond.shape[0], len(skybright)))
            skybrights[i,:] = skybright 
    
        wave = specsim_wave.value 

        # write to file 
        fsb = h5py.File(f_skybright, 'w')
        for ik, k in enumerate(['airmass', 'seeing', 'exptime', 'moonfrac', 'moonalt', 'moonsep', 'sunalt', 'sunsep']):
            fsb.create_dataset(k, data=obscond[:,ik]) 
        # save sky brightnesses
        fsb.create_dataset('wave', data=wave) 
        fsb.create_dataset('skybrightness', data=skybrights) 
        fsb.close() 

    if validate: 
        fig = plt.figure(figsize=(15,5))
        sub = fig.add_subplot(111)
        for i in range(skybrights.shape[0]):
            sub.plot(wave, skybrights[i,:], lw=1, c='C'+str(i), label=r'$i_{\rm obs}=%i$' % (i+1)) 
        sub.legend(loc='upper right', frameon=True, fontsize=20) 
        sub.set_xlabel(r'Wavelenght [$\AA$]', fontsize=25) 
        sub.set_xlim([3600., 9800.]) 
        sub.set_ylabel(r'Sky Surface Brightness', fontsize=25) 
        sub.set_ylim([0., 25.]) 
        fig.savefig(f_skybright.rsplit('.hdf5',1)[0]+'.png', bbox_inches='tight') 
    return wave, skybrights 


if __name__=="__main__":
    #_ = obs_condition(sampling='spacefill', validate=True)
    # save observing sky brightnesses
    #_ = obs_SkyBrightness(sampling='spacefill', validate=True)

    # load test set gal ids 
    galids = testGalIDs()
    for iobs in [0]: #range(1,8): 
        for ii, galid in enumerate(np.unique(galids)): 
            Lgal_BGSnoiseSpec(galid, iobs, lib='bc03', obs_sampling='spacefill', dust=False, overwrite=True, validate=True)
            Lgal_BGSnoiseSpec(galid, iobs, lib='bc03', obs_sampling='spacefill', dust=True, overwrite=True, validate=True)

    #mFF_LGal_nodust(imf='chabrier', iobs=0, obs_sampling='spacefill')