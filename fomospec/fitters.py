import os
import h5py
import time
import warnings 
import numpy as np 
import pandas as pd
import multiprocessing
from astropy import units as U
from astropy.cosmology import Planck13 as cosmo
from estimations_3d import estimation

# --- prospector --- 
import prospect as Prospect
# --- firefly ---
import StellarPopulationModel as spm
from StellarPopulationModel import trylog10
from firefly_library import airtovac, convert_chis_to_probs, light_weights_to_mass, calculate_averages_pdf
from firefly_dust import hpf, unred, determine_attenuation
from firefly_instrument import downgrade
from firefly_fitter import fitter, newFitter
# --- fomospec ---
from . import util as UT


dict_imfs = {'cha': 'Chabrier', 'ss': 'Salpeter', 'kr': 'Kroupa'}

class Prospector(object): 
    ''' class object for spectral fitting using prospector
    '''
    def __init__(self, zcontinuous=1, dust_type=0, sfh=4, add_burst=False, add_neb=True, **kwargs): 
        from prospect.models import priors
        from prospect.models.templates import TemplateLibrary

        self.zcontinuous = zcontinuous

        self.TemplateLibrary = TemplateLibrary
        
        model_params = self.TemplateLibrary["parametric_sfh"]
        
        # default priors for free parameters
        model_params["mass"]["prior"] = priors.LogUniform(
                mini=10**kwargs.get('logM_min', 9), 
                maxi=10**kwargs.get('logM_max', 11))
        model_params["logzsol"]["prior"] = priors.TopHat(
                mini=kwargs.get('logZsol_min', -1.), 
                maxi=kwargs.get('logZsol_max', 0.1))
        model_params["tage"]["prior"] = priors.TopHat(
                mini=kwargs.get('tage_min', 1.), 
                maxi=kwargs.get('tage_max', 13.6))

        # dust model 
        model_params['dust_type']['init'] = dust_type
        if dust_type == 0: 
            # dust2 describes the attenuation of old stellar light (Av)
            model_params["dust2"]["prior"] = priors.TopHat(
                    mini=kwargs.get('dust2_min', 0.), 
                    maxi=kwargs.get('dust2_max', 2.))
        else: 
            raise NotImplementedError
        # sfh model
        model_params['sfh']['init'] = sfh 
        if sfh == 4: # delayed tau model 
            # tau defines e-folding time for the SFH, in Gyr.
            model_params["tau"]["prior"] = priors.LogUniform(
                    mini=kwargs.get('tau_min', 1e-1), 
                    maxi=kwargs.get('tau_max', 1e2))
        else: 
            raise NotImplementedError
        
        # Add burst parameters (fixed to zero by default)
        model_params.update(TemplateLibrary["burst_sfh"])
        if add_burst: 
            pass
            # continue implementing 
            # continue implementing 
            # continue implementing 
            # continue implementing 
            # continue implementing 
            # continue implementing 


        # Add dust emission parameters (fixed)
        model_params.update(TemplateLibrary["dust_emission"])
     
        # Add nebular emission parameters and turn nebular emission on
        if add_neb:
            model_params.update(TemplateLibrary["nebular"])
        
        self.model_params = model_params
        self.sps = None 
        self.sedmodel = None 

    def model(self, lam, theta, zred, filters=None): 
        ''' wrapper for fsps based models. Given observed(?) wavelengths, 
        theta, and redshift return spectra, photometry, and mfrac. 
        '''
        if self.sps is None: self._loadSPS()
        if self.sedmodel is None: self._loadSEDmodel(zred=zred)
        if self.model_params['zred']['init'] != zred: 
            warnings.warn("there's a redshift mismatch between the stored model.") 
        
        obs = {}
        if filters is not None: 
            from sedpy.observate import load_filters
            obs['filters'] = load_filters(filters)
            mock["phot_wave"] = [f.wave_effective for f in mock["filters"]]
        else: 
            obs['filters'] = None 
        obs['wavelength'] = lam 
        # for spectrophotometry add in sedmodel.spe_calibration parameters
        # see https://github.com/bd-j/prospector/blob/master/prospect/models/sedmodel.py
        
        flux, phot, mfrac = self.sedmodel.mean_model(theta, obs=obs, sps=self.sps) # in maggies
        flux *= 1e17 * UT.c_light() / lam**2 * (3631. * UT.jansky_cgs()) # convert to 10^-17 ergs/s/cm^2/Ang
        return flux, phot, mfrac

    def dynesty_spec(self, lam, flux, flux_noise, zred, mask=False, N_angstrom_masked=20., 
            nested=True, bound='multi', sample='unif', nlive_init=100, nlive_batch=100, 
            maxcall_init=None, maxcall=None, write=True, output_file=None, silent=False): 
        ''' observed-frame wavelength and flux (in 10^-17 ergs/s/cm^2/Ang). There's also an 
        option to mask the emission line regions with the `mask` kwargs. 
        '''
        if write and output_file is None: raise ValueError 

        from prospect import fitting #from prospect.likelihood import lnlike_spec
        import dynesty 
        from dynesty.dynamicsampler import stopping_function, weight_function
        # observations 
        obs = {} 
        obs['wavelength'] = lam 
        obs['spectrum'] = flux 
        if not mask:
            obs['mask'] = np.ones(len(lam)).astype(bool)
        else: 
            w_lines = np.array([3728., 4861., 5007., 6564.]) * (1. + zred) 
            
            lines_mask = np.ones(len(lam)).astype(bool) 
            for wl in w_lines: 
                inemline = ((lam > wl - N_angstrom_masked) & (lam < wl + N_angstrom_masked))
                lines_mask = (lines_mask & ~inemline)
            obs['mask'] = lines_mask 

        if flux_noise is not None: 
            obs['unc'] = flux_noise 
        else: 
            obs['unc'] = np.ones(len(lam))
        
        # load SPS and SED model
        if self.sps is None: self._loadSPS()
        if self.sedmodel is None: self._loadSEDmodel(zred=zred)

        def lnPost(tt, nested=nested): 
            # Calculate prior probability and return -inf if not within prior
            # Also if doing nested sampling, do not include the basic priors,
            # since the drawing method already includes the prior probability
            lnp_prior = self.sedmodel.prior_product(tt, nested=nested)
            if not np.isfinite(lnp_prior):
                return -np.infty

            wave = obs['wavelength']
            # model(theta) flux [10^-17 ergs/s/cm^2/Ang]
            spec, _, _ = self.model(wave, tt, zred, filters=None)

            # Calculate likelihoods
            #lnp_spec = lnlike_spec(spec, obs=obs, spec_noise=None)
            mask = obs['mask'] 
            delta = (obs['spectrum'] - spec)[mask] 
            var = (obs['unc'][mask])**2
            lnp_spec = -0.5 * (delta**2/var).sum() 
            return lnp_prior + lnp_spec
        
        def prior_transform(u): 
            return self.sedmodel.prior_transform(u)
        
        # dynesty Fitter parameters
        dyn_params = {'nested_bound': bound, # bounding method
                      'nested_sample': sample, # sampling method
                      'nested_nlive_init': nlive_init,
                      'nested_nlive_batch': nlive_batch,
                      'nested_bootstrap': 0,
                      'nested_dlogz_init': 0.05,
                      'nested_maxcall_init': maxcall_init, 
                      'nested_maxcall': maxcall, 
                      'nested_weight_kwargs': {"pfrac": 1.0},
                      'nested_stop_kwargs': {"post_thresh": 0.1}}
        run_params = self.model_params.copy() 
        run_params.update(dyn_params)
    
        tstart = time.time()  # time it
        out = fitting.run_dynesty_sampler(
                lnPost, 
                prior_transform, 
                self.sedmodel.ndim, 
                stop_function=stopping_function, 
                wt_function=weight_function,
                **run_params)
        duration = time.time() - tstart
        if not silent: print("dynesty sampler took %f sec" % duration) 

        if not write: 
            return out  
        else: 
            from prospect.io import write_results
            if not silent: print("Writing to %s" % output_file)
            self.model_params["outfile"] = output_file 

            write_results.write_hdf5(output_file, run_params, self.sedmodel, 
                    obs, out, None, tsample=duration)
            return None

    def read_dynesty(self, fname):  
        ''' Read in output from dynesty_spec
        '''
        f = h5py.File(fname, 'r') 
        # observation that's being fit 
        obvs = {} 
        for k in f['obs'].keys(): 
            obvs[k] = f['obs'][k].value

        output = {} 
        for k in f['sampling'].keys(): 
            output[k] = f['sampling'][k].value 
        f.close()
        return output, obvs

    def emcee_spec(self, lam, flux, flux_noise, zred, mask=False, N_angstrom_masked=20., 
            min_method='levenberg_marquardt', write=True, output_file=None, silent=False): 
        ''' infer parameters using emcee
        '''
        if write and output_file is None: raise ValueError 
        from prospect import fitting
        from scipy.optimize import least_squares, minimize
        
        if self.sps is None: self._loadSPS()
        if self.sedmodel is None: self._loadSEDmodel(zred=zred)
        # observations 
        obs = {} 
        obs['wavelength'] = lam 
        obs['spectrum'] = flux 
        if not mask:
            obs['mask'] = np.ones(len(lam)).astype(bool)
        else: 
            w_lines = np.array([3728., 4861., 5007., 6564.]) * (1. + zred) 
            
            lines_mask = np.ones(len(lam)).astype(bool) 
            for wl in w_lines: 
                inemline = ((lam > wl - N_angstrom_masked) & (lam < wl + N_angstrom_masked))
                lines_mask = (lines_mask & ~inemline)
            obs['mask'] = lines_mask 
        if flux_noise is not None: 
            obs['unc'] = flux_noise 
        else: 
            obs['unc'] = np.ones(len(lam))

        def lnPost(tt): 
            # Calculate prior probability and return -inf if not within prior
            # Also if doing nested sampling, do not include the basic priors,
            # since the drawing method already includes the prior probability
            lnp_prior = self.sedmodel.prior_product(tt)
            if not np.isfinite(lnp_prior):
                return -np.infty

            wave = obs['wavelength']
            # model(theta) flux [10^-17 ergs/s/cm^2/Ang]
            spec, _, _ = self.model(wave, tt, zred, filters=None)

            # Calculate likelihoods
            #lnp_spec = lnlike_spec(spec, obs=obs, spec_noise=None)
            mask = obs['mask'] 
            delta = (obs['spectrum'] - spec)[mask] 
            var = (obs['unc'][mask])**2
            lnp_spec = -0.5 * (delta**2/var).sum() 
            return lnp_prior + lnp_spec

        def chivec(tt): 
            """A version of lnprobfn that returns the simple uncertainty
            normalized residual instead of the log-posterior, for use with
            least-squares optimization methods like Levenburg-Marquardt.

            It's important to note that the returned chi vector does not
            include the prior probability.
            """
            lnp_prior = self.sedmodel.prior_product(tt)
            if not np.isfinite(lnp_prior):
                return -np.infty

            wave = obs['wavelength']
            # model(theta) flux in [10^-17 ergs/s/cm^2/Ang]
            try: 
                spec, _, _ = self.model(wave, tt, zred, filters=None)
            except ValueError: 
                return -np.infty

            mask = obs['mask'] 
            delta = (obs['spectrum'] - spec)[mask] 
            chi = delta/(obs['unc'][mask])
            return chi 
    
        t_min0 = time.time()         
        # minimization to get initial theta 
        min_params = {
                'nmin': 5, 
                'ftol': 3e-16, 
                'maxfev': 5000, 
                'xtol': 3e-16, 
                'min_method': min_method
                } 
        if not silent: print('initial_theta', self.sedmodel.initial_theta)
        # draw initial values from the (1d, separable, independent) priors 
        # for each parameter.
        pinitial = fitting.minimizer_ball(self.sedmodel.initial_theta.copy(), 
                min_params['nmin'], self.sedmodel)
        if not silent: print('pinitial', pinitial)
        guesses = []
        for i, pinit in enumerate(pinitial): #loop over initial guesses
            res = least_squares(chivec, np.array(pinit), method='lm', x_scale='jac',
                    xtol=min_params["xtol"], ftol=min_params["ftol"], 
                    max_nfev=min_params["maxfev"])
            if not silent: print('res', i, res)
            guesses.append(res)

        ## Calculate chi-square of the results, and choose the best one
        ## fitting.reinitialize moves the parameter vector away from edges of the prior.
        chisq = [np.sum(r.fun**2) for r in guesses]
        best = np.argmin(chisq)
        theta_best = fitting.reinitialize(guesses[best].x, self.sedmodel,
                                          edge_trunc=min_params.get('edge_trunc', 0.1))
        if not silent: 
            print('minimum chisq theta', theta_best) 
            t_min = (time.time() - t_min0)/60.
            print('minimization takes %f' % t_min)

        # run emcee
        emcee_params = {
                'nwalkers': 128,    # number of emcee walkers
                'niter': 512,       # number of iterations in each round of burn-in 
                # After each round, the walkers are reinitialized based on the 
                # locations of the highest probablity half of the walkers.
                'nburn': [16, 32, 64],
                # The following number controls how often the chain is written to disk. 
                # This can be useful to make sure that not all is lost if the code dies 
                # during a long MCMC run. It ranges from 0 to 1; the current chains will 
                # be written out every `interval` * `niter` iterations. The default is 1, 
                # i.e. only write out at the end of the run.
                'interval': 0.25 # write out after every 25% of the sampling is completed.
                } 
        run_params = self.model_params.copy() 
        run_params.update(emcee_params)

        initial_center = theta_best.copy()
        
        if not silent: t_mc0 = time.time()         
        hfile = h5py.File(output_file, 'a') 

        out = fitting.run_emcee_sampler(lnPost, initial_center, self.sedmodel,
                hdf5=hfile, **run_params)

        if not silent: 
            t_mc = (time.time() - t_mc0)/60.
            print('emcee takes %f' % t_min)
        if not write: 
            return out  
        else: 
            from prospect.io import write_results
            if not silent: print("Writing to %s" % output_file)

            esampler, burn_loc0, burn_prob0 = out
            write_results.write_hdf5(hfile, run_params, self.sedmodel, obs, 
                         esampler, guesses,
                         sampling_initial_center=initial_center,
                         post_burnin_center=burn_loc0,
                         post_burnin_prob=burn_prob0)
            return None

    def read_emcee(self, fname): 
        ''' read in output from emcee_spec 
        '''
        f = h5py.File(fname, 'r') 
        f.close() 

    def _loadSPS(self):  
        from prospect.sources import CSPSpecBasis
        self.sps = CSPSpecBasis(zcontinuous=self.zcontinuous)
        return None 

    def _loadSEDmodel(self, zred=None): 
        # instantiate the model using self.model_params parameter specifications
        from prospect.models import SedModel
        if zred is not None: 
            self.model_params["zred"]["isfree"] = False 
            self.model_params["zred"]["init"] = zred
        self.sedmodel = SedModel(self.model_params)
        return None 


class myFirefly(object): 
    ''' my own re-written version of firefly  
    '''
    def __init__(self, specobj, cosmo, 
            model='m11', model_lib='MILES', imf='kr', downgrade_models=True, 
            dust_corr=True, dust_law='calzetti', max_ebv=1.5, num_dust_vals=200, dust_smoothing_length=200,
            age_lim=[0, 14.], # in Gyrs
            logZ_lim=[-0.1,0.1], 
            wave_lim=[0,1e5], 
            data_wave_medium='vacuum', 
            max_iterations=10, fit_per_iteration_cap=1000, pdf_sampling=300, 
            output=None):
        self.cosmo = cosmo      # save cosmology 
        self.specObj = specobj  # spectra object
        if output is not None: 
            self.f_output = output  
        else: 
            self.f_output = None 

        # sets the models 
        self.model = model # m11 / m09
        self.model_lib = model_libs
        self.downgraded_models = downgraded_models
        # get library wavelength resolution 
        self.deltal_lib = self._check_model_lib(self.model, self.model_lib, downgrade_models=downgrade_models) 
        # sets the Initial mass function
        self.imf = imf
        
        # dust stuff (pretty much untouched from original firefly) 
        self.dust_corr = dust_corr 
        self.dust_law = dust_law
        self.max_ebv = max_ebv
        self.num_dust_vals = num_dust_vals
        self.dust_smoothing_length = dust_smoothing_length
    
        self.age_lim = age_lim # in Gyrs because I'm not fucking insane
        self.logZ_lim = logZ_lim # log metallicity 
        self.wave_lim = wave_lim # in angstrom

        # Specific fitting options
        self.max_iterations = max_iterations
        self.fit_per_iteration_cap = fit_per_iteration_cap
        # Sampling size when calculating the maximum pdf (100=recommended)
        self.pdf_sampling = pdf_sampling
        # Default is air, unless manga is used
        self.data_wave_medium = data_wave_medium
        
        self.vdisp_round = int(round(self.specObj.vdisp/5.0)*5.0) # rounding vDisp for the models 

    def get_model(self, vdisp, wave_instrument, r_instrument, ebv_mw, silent=True):
        ''' Retrieves all relevant model files, in their downgraded format.
        If they aren't downgraded to the correct resolution / velocity dispersion,
        takes the base models in their native form and converts to downgraded files.

        descriptions of SSP in http://www-astro.physics.ox.ac.uk/~maraston/SSPn/SED/README_Mar05
        ''' 
        dir_sp = os.environ['STELLARPOPMODELS_DIR']+'/data'

        model_files = []
        if self.model == 'm11':
            str_model = '.'.join([self.model_lib, self.imf]) 
            if self.downgraded_models :
                if model_used in ['MILES_UVextended', 'MILES_revisedIRslope']:
                    model_path = '/'.join([dir_sp, 
                        'SSP_M11_MILES_downgraded','ssp_M11_'+str_model])
                else: 
                    model_path = '/'.join([dir_sp,
                        'SSP_M11_'+ model_used + '_downgraded', 'ssp_M11_'+str_model])
            else:
                if model_used in ['MILES_UVextended', 'MILES_revisedIRslope']:
                    model_path = '/'.join([dir_sp, 
                        'SSP_M11_MILES', 'ssp_M11_'+str_model])
                else:
                    model_path = '/'.join([dir_sp, 
                        'SSP_M11_'+model_used ,'ssp_M11_'+str_model])
            
            z_strs = np.array(['z001', 'z002', 'z004', 'z0001.bhb', 'z0001.rhb', 
                'z10m4.bhb', 'z10m4.rhb', 'z10m4', 'z-0.6', 'z-0.9', 'z-1.2', 'z-1.6', 'z-1.9']) 
            z = np.array([0.5, 1.0, 2.0, 10**-1.301, 10**-1.302, 10**-2.301, 
                10**-2.302, 10**-2.300, 10**-0.6, 10**-0.9, 10**-1.2, 10**-1.6, 10**-1.9]) 
            
        elif self.model == 'm09':
            if self.downgraded_models:
                model_path = ''.join([dir_sp, 'UVmodels_Marastonetal08b_downgraded'])
            else:
                model_path = ''.join([dir_sp, 'UVmodels_Marastonetal08b'])

            z_strs = np.array(['z001', 'z002', 'z004', 'z0001']) 
            z = np.array([10**-0.3, 1.0, 10**0.3, 10**-1.3]) 
        else: 
            raise ValueError
    
        # keep only files with metallicities within the range 
        inZlim = (z > 10**logZ_lim[0]) & (z < 10**logZ_lim[1]) 
        metal = z[inZlim] 
        metal_files = [model_path+zstr for zstr in z_strs[inZlim]]
        if not silent: print('metal files included %s from %s' % (', '.join(metal_files), model_path))

        # constructs the model array
        flux_model, age_model, metal_model = [],[],[]
        for met, f_metal in zip(metal, metal_files):
            model_age, model_wave, model_flux = np.loadtxt(f_metal, unpack=True, usecols=[0,2,3]) 

            uniq_age = np.unique(model_age) # Gyr 
            for age in uniq_age:
                if (age < age_lim[0]) or (age > age_lim[1]): # outside of age limit 
                    continue

                wave = model_wave[model_age == age] 
                flux = model_flux[model_age == age]
                    
                # converts to air wavelength
                if self.data_wave_medium == 'vacuum':
                    wavelength = airtovac(wave)
                else:
                    wavelength = wave 

                #if self.downgrade_models:
                # downgrades the model to match the data resolution 
                #mf = downgrade(wavelength, flux, deltal, self.vdisp_round, wave_instrument, r_instrument)
                #else: 
                mf = flux.copy() 

                # Reddens the models
                if ebv_mw != 0:
                    attenuations = unred(wavelength,ebv=0.0-ebv_mw)
                    flux_model.append(mf*attenuations)
                else:
                    flux_model.append(mf)
                age_model.append(age) # Gyr
                metal_model.append(met) 
        return wavelength, np.array(flux_model), np.array(age_model), np.array(metal_model)
	
    def Fit(self, wave, flux, err, zred, mask=None, vdisp=None, r_instrument=None, ebv_mw=None, silent=True):
        ''' Once the data and models are loaded, then execute this function to 
        find the best model. It loops overs the models to be fitted on the data:
        #. gets the models
        #. matches the model and data to the same resolution
        #. normalises the spectra
        '''
        if vdisp is None: 
            vdisp = 70. 
        if r_instrument is None: 
            r_instrument = self._r_instrument_default(wave)
        if ebv_mw is None: 
            ebv_mw = 0.

        wave_rest = wave / (1. + zred) 

        if not silent: print("1. getting the models")
        model_wave_int, model_flux_int, model_age, model_metal = self.get_model(
                vdisp, 
                wave_rest, 
                r_instrument, 
                ebv_mw)

        if not silent: print("2. match model and data resolution")
        wave, data_flux, error_flux, model_flux_raw = self._match_data_models(
                wave_rest, 
                flux, 
                err, 
                model_wave_int, 
                model_flux_int, 
                mask=mask, 
                silent=silent)

        if self.dust_corr:
            if not silent: print("Dust correction")
            # determining attenuation curve through HPF fitting, 
            # apply attenuation curve to models and renormalise spectra
            best_ebv, attenuation_curve = determine_attenuation(wave, data_flux, error_flux, 
                    model_flux, self, model_age, model_metal)
            model_flux_atten = np.zeros(model_flux_raw.shape)
            for m in range(len(model_flux_raw)):
                model_flux_atten[m] = attenuation_curve * model_flux_raw[m]

            data_median = np.median(data_flux)
            model_median = np.median(model_flux_atten, axis=1) 
            
            mass_factor = data_median/model_median
            model_flux = (model_flux_atten.T / model_median).T * data_median
        else: 
            if not silent: print("Normalize the models to the median data flux")
            data_median = np.median(data_flux)
            model_median = np.median(model_flux_raw, axis=1) 

            mass_factor = data_median/model_median
            model_flux = (model_flux_raw.T / model_median).T * data_median

        if not silent: print("Fit model to the data")
        # this is supposed to be the essential ingredient of firefly 
        # the code is a giant mess with nested functions and classes 
        # and global variables everywhere. 
        light_weights, chis, branch = fitter(wave, data_flux, error_flux, model_flux, self)

        if not silent: print("mass weighted SSP contributions")
        # 5. Get mass-weighted SSP contributions using saved M/L ratio.
        unnorm_mass = light_weights * mass_factors
        mass_weights = unnorm_mass / np.sum(unnorm_mass, axis=0) 

        if not silent: print("chis into probabilities")
        self.dof = len(wave)
        probs = convert_chis_to_probs(chis, self.dof)
        self.d_lum = self.cosmo.luminosity_distance(zred).to(U.cm).value
                        
        if not silent: print("Calculating average properties and outputting")
        averages = calculate_averages_pdf(probs, 
                light_weights, 
                mass_weights, 
                unnorm_mass, 
                model_age, 
                model_metal, 
                self.pdf_sampling, 
                self.d_lum, 
                1e-17)
        
        unique_ages = np.unique(age_model)
        marginalised_age_weights = np.zeros(unique_ages.shape)
        marginalised_age_weights_int = np.sum(mass_weights.T, 1)
        for age in unique_ages:
            marginalised_age_weights[ua] = np.sum(marginalised_age_weights_int[age_model == age])

        best_fit_index = [np.argmin(chis)]
        best_fit = np.dot(light_weights[best_fit_index], model_flux)[0]

        bf_mass = (mass_weights[best_fit_index] > 0)[0]
        bf_light = (light_weights[best_fit_index] > 0)[0]
        mass_per_ssp = unnorm_mass[best_fit_index[0]][bf_mass] * 1e-17 * 4 * np.pi * self.d_lum**2.0 
        age_per_ssp = age_model[bf_mass]
        metal_per_ssp = metal_model[bf_mass]
        weight_mass_per_ssp = mass_weights[best_fit_index[0]][bf_mass]
        weight_light_per_ssp = light_weights[best_fit_index[0]][bf_light]
        order = np.argsort(-weight_light_per_ssp)

        final_ML_totM, final_ML_alive, final_ML_wd, final_ML_ns, final_ML_bh, final_ML_turnoff, final_gas_fraction = self._get_massloss_factors(imf, mass_per_ssp, age_per_ssp, metal_per_ssp)

        # Calculate the total mass loss from all the SSP contributions.
        combined_ML_totM = np.sum(final_ML_totM)
        combined_ML_alive = np.sum(final_ML_alive)
        combined_ML_wd = np.sum(final_ML_wd)
        combined_ML_ns = np.sum(final_ML_ns)
        combined_ML_bh = np.sum(final_ML_bh)		
        combined_gas_fraction = np.sum(mass_per_ssp - final_ML_totM)
       
        ff_fit = {} 
        ff_fit['wavelength'] = wave
        ff_fit['flux'] = data_flux
        ff_fit['flux_err'] = error_flux 
        ff_fit['flux_model'] = best_fit 

        ff_prop = {} 
        ff_prop['redshift'] = zred
        ff_prop['imf'] = self.imf 
        ff_prop['model'] = self.model_lib 
        ff_prop['model_lib'] = self.model 
        ff_prop['prior_age'] = self.age_lim
        ff_prop['prior_logZ'] = self.logZ_lim
    
        # age and metallicity 
        for w in ['light', 'mass']: 
            ff_prop['age_'+w+'W'] = averages[w+'_age'] # Gyr
            ff_prop['logZ_'+w+'W'] = np.log10(averages[w+'_metal'])
            for i_q in np.arange(1,4).astype(str): 
                ff_prop['age_'+w+'W_up_'+i_q+'sig'] = averages[w+'_age_'+i_q+'_sig_plus']
                ff_prop['age_'+w+'W_up_'+i_q+'sig'] = averages[w+'_age_'+i_q+'_sig_plus']
                ff_prop['logZ_'+w+'W_up_'+i_q+'sig'] = np.log10(averages[w+'_metallicity_'+i_q+'_sig_plus'])
                ff_prop['logZ_'+w+'W_up_'+i_q+'sig'] = np.log10(averages[w+'_metallicity_'+i_q+'_sig_plus'])

        # total stellar mass 
        ff_prop['logM_total'] = np.log10(averages['stellar_mass'])
        for i_q in np.arange(1,4).astype(str):  
            ff_prop['logM_total_up_'+i_q+'sig'] = np.log10(averages['stellar_mass_'+i_q+'_sig_plus'])
            ff_prop['logM_total_low_'+i_q+'sig'] = np.log10(averages['stellar_mass_'+i_q+'_sig_minus'])

        ff_prop['logM_stellar'] = np.log10(combined_ML_alive+combined_ML_wd+combined_ML_ns+combined_ML_bh)
        ff_prop['logM_living_stars'] = trylog10(combined_ML_alive)
        ff_prop['logM_remnant'] = trylog10(combined_ML_wd+combined_ML_ns+combined_ML_bh)
        ff_prop['logM_remnant_in_whitedwarfs'] = trylog10(combined_ML_wd)
        ff_prop['logM_remnant_in_neutronstars'] = trylog10(combined_ML_ns)
        ff_prop['logM_remnant_blackholes'] = trylog10(combined_ML_bh)
        ff_prop['logM_mass_of_ejecta'] = trylog10(combined_gas_fraction)
        
        ff_prop['n_ssp'] =len(order)
        for i in range(len(order)): # quantities per SSP
            ff_prop['logM_total_ssp_'+str(iii)] = np.log10(mass_per_ssp[order][iii])
            ff_prop['logM_stellar_ssp_'+str(iii)] = np.log10(final_ML_alive[order][iii]+final_ML_wd[order][iii]+final_ML_ns[order][iii]+final_ML_bh[order][iii])
            ff_prop['logM_living_stars_ssp_'+str(iii)] = np.log10(final_ML_alive[order][iii])	
            ff_prop['logM_remnant_ssp_'+str(iii)] = np.log10(final_ML_wd[order][iii]+final_ML_ns[order][iii]+final_ML_bh[order][iii])
            ff_prop['logM_remnant_in_whitedwarfs_ssp_'+str(iii)] = np.log10(final_ML_wd[order][iii])
            ff_prop['logM_remnant_in_neutronstars_ssp_'+str(iii)] = np.log10(final_ML_ns[order][iii])
            ff_prop['logM_remnant_in_blackholes_ssp_'+str(iii)] = np.log10(final_ML_bh[order][iii])
            ff_prop['logM_mass_of_ejecta_ssp_'+str(iii)] = np.log10(mass_per_ssp[order][iii] - final_ML_totM[order][iii])
            ff_prop['age_ssp_'+str(iii)] = age_per_ssp[order][iii] # Gyr
            ff_prop['logZ_ssp_'+str(iii)] = np.log10(metal_per_ssp[order][iii])
            ff_prop['logSFR_ssp_'+str(iii)] = np.log10(mass_per_ssp[order][iii]/age_per_ssp[order][iii])	
            ff_prop['weightMass_ssp_'+str(iii)] = weight_mass_per_ssp[order][iii]
            ff_prop['weightLight_ssp_'+str(iii)] = weight_light_per_ssp[order][iii]
        
        if self.dust_corr: 
            ff_prop['ebv'] = best_ebv

        if self.f_output is None:  
            if not silent: print('write to %s' % self.f_output) 
            f = h5py.File(self.f_output, 'w') 
            for k in ff_fit.keys():
                f.create_dataset(k, data=ff_fit[k]) 

            for k in ff_prop.keys(): 
                f.attr[k] = ff_prop[k] 
            
        return ff_fit, ff_prop

    def _get_massloss_factors(self, imf, mass_per_ssp, age_per_ssp, metal_per_ssp): 
        # Gets the mass loss factors.
       dict_imfs = {'ss': 'salpeter', 'cha': 'chabrier', 'kr': 'kroupa'} 
       f_ml = '/'.join([os.environ['STELLARPOPMODELS_DIR'],'data','massloss_'+str(dict_imfs[imf])+'.txt'])
       ML_metallicity, ML_age, ML_totM, ML_alive, ML_wd, ML_ns, ML_bh, ML_turnoff = np.loadtxt(f_ml, 
               unpack=True, skiprows=2)

       # First build the grids of the quantities. Make sure they are in linear units.                  
       estimate_ML_totM = estimation(10**ML_metallicity, ML_age, ML_totM) 
       estimate_ML_alive = estimation(10**ML_metallicity, ML_age, ML_alive) 
       estimate_ML_wd = estimation(10**ML_metallicity, ML_age, ML_wd) 
       estimate_ML_ns = estimation(10**ML_metallicity, ML_age, ML_ns) 
       estimate_ML_bh = estimation(10**ML_metallicity, ML_age, ML_bh)
       estimate_ML_turnoff = estimation(10**ML_metallicity, ML_age, ML_turnoff) 

       # Now loop through SSPs to find the nearest values for each.
       final_ML_totM, final_ML_alive, final_ML_wd, final_ML_ns = [], [], [], [] 
       final_ML_bh, final_ML_turnoff, final_gas_fraction = [], [], []
       for i in range(len(age_per_ssp)):
           new_ML_totM = estimate_ML_totM.estimate(metal_per_ssp[i], age_per_ssp[i])
           new_ML_alive = estimate_ML_alive.estimate(metal_per_ssp[i], age_per_ssp[i])
           new_ML_wd = estimate_ML_wd.estimate(metal_per_ssp[i], age_per_ssp[i])
           new_ML_ns = estimate_ML_ns.estimate(metal_per_ssp[i], age_per_ssp[i])
           new_ML_bh = estimate_ML_bh.estimate(metal_per_ssp[i], age_per_ssp[i])
           new_ML_turnoff = estimate_ML_turnoff.estimate(metal_per_ssp[i], age_per_ssp[i]) 

           final_ML_totM.append(mass_per_ssp[i]*new_ML_totM)
           final_ML_alive.append(mass_per_ssp[i]*new_ML_alive)
           final_ML_wd.append(mass_per_ssp[i]*new_ML_wd)
           final_ML_ns.append(mass_per_ssp[i]*new_ML_ns)
           final_ML_bh.append(mass_per_ssp[i]*new_ML_bh)
           final_ML_turnoff.append(mass_per_ssp[i]*new_ML_turnoff)
           final_gas_fraction.append(mass_per_ssp[i] - new_ML_totM)

        return np.array(final_ML_totM), np.array(final_ML_alive), np.array(final_ML_wd), np.array(final_ML_ns), np.array(final_ML_bh), np.array(final_ML_turnoff), np.array(final_gas_fraction)

    def _match_data_models(self, w_d, flux_d, err_d, w_m, flux_m, mask=None, silent=True): 
        '''
        '''
        if mask is None:   
            mask = np.ones(len(w_d)).astype(bool) 
        n_models = flux_m.shape[0]
        w_min, w_max = w_d[mask].min(), w_d[mask].max()
        if not silent: print("%f < wavelength < %f" % (w_min, w_max))

        mask_m = ((w_m <= w_min) & (w_m >= w_max))
        if np.sum(mask_m) == 0: 
            raise ValueError("outside of model coverage")
    
        w_m = w_m[mask_m]
        flux_m = flux_m[:,mask_m] 

        w_d = w_d[mask]
        flux_d = flux_d[mask] 
        err_d =  err_d[mask]

        if np.sum(mask_m) >= np.sum(mask): # model has higher wavelength resolution
            w_d_mid = 0.5 * (w_d[1:] + w_d[:-1]) + w_d.min()*0.0000000001

            matched_model = np.zeros((n_models, len(w_d_mid)-1)) 
            for i in range(n_models):
                flux_m_bounds = np.interp(w_d_mid, w_m, flux_m[i])

                combined_wave = np.concatenate((w_m, w_d_mid))
                combined_flux = np.concatenate((flux_m[i], flux_m_bounds))
                sort_indices = np.argsort(combined_wave)
               
                combined_wave = combined_wave[sort_indices]
                combined_flux = combined_flux[sort_indices]
                
                boundary_indices = np.searchsorted(combined_wave, w_d_mid) 

                n_combo = len(combined_flux) 
                for l in range(len(boundary_indices) - 1):
                    if boundary_indices[l + 1] >= n_combo:
                        matched_model[m][l] = matched_model[m][l - 1]
                    else:
                        matched_model[m][l] = np.trapz(
                                combined_flux[boundary_indices[l]:boundary_indices[l+1]+ 1], 
                                x=combined_wave[boundary_indices[l]:boundary_indices[l+1]+1]
                                )
                        matched_model[m][l] /= (
                                combined_wave[boundary_indices[l+1]]-combined_wave[boundary_indices[l]])
            matched_wave = w_d[1:-1]
            matched_data = flux_d[1:-1]
            matched_error = err_d[1:-1]
        else:
            w_m_mid = 0.5 * (w_m[1:]+w_m[:-1]) + w_m.min()*0.0000000001
            boundaries = np.searchsorted(w_d, w_m_mid)

            flux_d_bounds   = np.interp(w_m_mid, w_d, flux_d)
            err_bounds 	    = np.interp(w_m_mid, w_d, err_d)
            combined_wave_int 	= np.concatenate((w_d, w_m_mid))
            combined_flux_int 	= np.concatenate((flux_d, flux_d_bounds))
            combined_err_int 	= np.concatenate((err_d, err_d_bounds))
            sort_indices        = np.argsort(combined_wave_int)
        
            combined_wave       = np.sort(combined_wave_int)
            boundary_indices 	= np.searchsorted(combined_wave,bisect_model)
            combined_flux 	= combined_flux_int[sort_indices]
            combined_error 	= combined_error_int[sort_indices]

            matched_data    = np.zeros(len(boundary_indices) - 1), 
            matched_error   = np.zeros(len(boundary_indices) - 1)

            n_combo = len(combined_flux)
            for l in range(len(boundary_indices) - 1):
                if boundary_indices[l+1] >= n_combo:
                    matched_data[l] 	= matched_data[l-1]
                    matched_error[l] 	= matched_error[l-1]
                else:
                    matched_data[l] = np.trapz(combined_flux[boundary_indices[l]:boundary_indices[l+1]+1], 
                            x=combined_wave[boundary_indices[l]:boundary_indices[l+1]+1])
                    matched_data[l] /= (combined_wave[boundary_indices[l+1]] - combined_wave[boundary_indices[l]])
                    matched_error[l] = np.trapz(combined_error[boundary_indices[l]:boundary_indices[l+1]+1], 
                            x=combined_wave[boundary_indices[l]:boundary_indices[l+1]+1])
                    matched_error[l] /= (combined_wave[boundary_indices[l+1]] - combined_wave[boundary_indices[l]])
            matched_wave = w_m[1:-1]
            matched_model = np.zeros((num_models,len(matched_wave)))
            for i in range(n_models):
                matched_model[i][:] = flux_m[i][1:-1]
        return matched_wave, matched_data, matched_error, matched_model

    def _check_model_lib(model, model_lib, downgrade_models=True):  
        ''' check model libraries 
        '''
        deltal_lib = []
        if model == 'm11':
            for m in model_lib:
                if m == 'MILES' or m == 'MILES_revisednearIRslope' or m == 'MILES_UVextended':
                    deltal_lib.append(2.55)
                elif m == 'STELIB':
                    deltal_lib.append(3.40)
                elif m == 'ELODIE':
                    deltal_lib.append(0.55)
                elif m == 'MARCS':
                    deltal_lib.append(0.1)
        elif model == 'bc03':
            if not np.array_equal(np.array(model_lib), np.array(['STELIB_BC03'])):
                raise ValueError('bc03 model only supports STELIB_BC03') 
            deltal_lib.append(3.00)
        elif model == 'm09':
            if not np.array_equal(np.array(model_lib), np.array(['M09'])):
                raise ValueError('bc03 model only supports M09') 
            if downgrade_models:
                deltal_lib.append(0.4)
            else:
                deltal_lib.append(3.6)
        return deltal_lib
    
    def _r_instrument_default(self, wave): 
        r_instrument = np.zeros(len(wave))
        low_w = (w < 6000.) 
        r_instrument[low_w] = (2270.0-1560.0)/(6000.0-3700.0) * wave[low_w] + 420.0 
        r_instrument[~low_w] = (2650.0-1850.0)/(9000.0-6000.0) * wave[~low_w] + 250.0 
        return r_instrument


class Firefly(spm.StellarPopulationModel): 
    def fit_models_to_data(self, silent=True): 
        ''' simplificaiton of the method in StellarPopulationModel 

        __original description__: Once the data and models are loaded, then 
        execute this function to find the best model. It loops overs the models 
        to be fitted on the data:
        #. gets the models
        #. matches the model and data to the same resolution
        #. normalises the spectra
        '''
        for mi, mod in enumerate(self.model_libs): # loop over the models
            for imf in self.imfs: # loop over the IMFs 
                # A. gets the models
                if not silent: print("getting the models")
                model_wave_int, model_flux_int, age, metal = self.get_model(
                        mod, # model 
                        imf, # IMF 
                        self.deltal_libs[mi],
                        self.specObs.vdisp, 
                        self.specObs.restframe_wavelength, 
                        self.specObs.r_instrument, 
                        self.specObs.ebv_mw)

                # B. matches the model and data to the same resolution
                if not silent: print("Matching model and data resolutions ")
                wave, data_flux, error_flux, model_flux_raw = match_data_models(
                        self.specObs.restframe_wavelength, 
                        self.specObs.flux, 
                        self.specObs.bad_flags, 
                        self.specObs.error, 
                        model_wave_int, 
                        model_flux_int, 
                        self.wave_limits[0], 
                        self.wave_limits[1], 
                        saveDowngradedModel=False)

                # C. normalises the models to the median value of the data
                if not silent: print("Normalising the amplitude of the models")
                model_flux, mass_factors = normalise_spec(data_flux, model_flux_raw)

                # 3. Corrects from dust attenuation
                if not silent: print("Correction from dust attentuation")
                if self.hpf_mode == 'on':
                    # 3.1. Determining attenuation curve through HPF fitting, apply 
                    # attenuation curve to models and renormalise spectra
                    if not silent: print("determining attentuation")
                    best_ebv, attenuation_curve = determine_attenuation(
                            wave, 
                            data_flux, 
                            error_flux, 
                            model_flux, 
                            self, 
                            age, 
                            metal)

                    #model_flux_atten = np.zeros(np.shape(model_flux_raw))
                    #for m in range(len(model_flux_raw)):
                    #    model_flux_atten[m] = attenuation_curve * model_flux_raw[m] 
                    model_flux_atten = attenuation_curve * model_flux_raw
                    model_flux, mass_factors = normalise_spec(data_flux, model_flux_atten)
                    # 4. Fits the models to the data
                    if not silent: print("fitter")
                    light_weights, chis, branch = self.Fitter(wave, data_flux, error_flux, model_flux)

                elif self.hpf_mode == 'hpf_only':
                    # 3.2. Uses filtered values to determing SP properties only."
                    smoothing_length = self.dust_smoothing_length
                    hpf_data    = hpf(data_flux)
                    hpf_models  = np.zeros(np.shape(model_flux))
                    for m in range(len(model_flux)):
                        hpf_models[m] = hpf(model_flux[m])

                    zero_dat = np.where( (np.isnan(hpf_data)) & (np.isinf(hpf_data)) )
                    hpf_data[zero_dat] = 0.0
                    for m in range(len(model_flux)):
                        hpf_models[m,zero_dat] = 0.0
                    hpf_error    = np.zeros(len(error_flux))
                    hpf_error[:] = np.median(error_flux)/np.median(data_flux) * np.median(hpf_data)
                    hpf_error[zero_dat] = np.max(hpf_error)*999999.9

                    best_ebv = 0.0
                    hpf_models, mass_factors = normalise_spec(hpf_data, hpf_models)
                    # 4. Fits the models to the data
                    if not silent: print("fitter")
                    light_weights, chis, branch = self.Fitter(wave, hpf_data, hpf_error, hpf_models)
                else: 
                    raise ValueError("hpf_mode has to be 'on' or 'hpf_only'")

                # 5. Get mass-weighted SSP contributions using saved M/L ratio.
                unnorm_mass, mass_weights = light_weights_to_mass(light_weights, mass_factors)

                if np.all(np.isnan(mass_weights)):
                    raise ValueError # tbhdu = self.create_dummy_hdu()

                # 6. Convert chis into probabilities and calculates all average properties and errors
                if not silent: print("Calculating average properties and outputting")
                self.dof = len(wave)
                probs = convert_chis_to_probs(chis, self.dof)
                dist_lum = self.cosmo.luminosity_distance(self.specObs.redshift).to(U.cm).value
                    
                averages = calculate_averages_pdf(probs, light_weights, mass_weights, unnorm_mass, age, metal, 
                        self.pdf_sampling, dist_lum, self.flux_units) 

                unique_ages 		        = np.unique(age)
                marginalised_age_weights        = np.zeros(np.shape(unique_ages))
                marginalised_age_weights_int    = np.sum(mass_weights.T,1)
                for ua in range(len(unique_ages)):
                    marginalised_age_weights[ua] = np.sum(marginalised_age_weights_int[np.where(age==unique_ages[ua])])

                best_fit_index = [np.argmin(chis)]
                best_fit = np.dot(light_weights[best_fit_index],model_flux)[0]
                # stores outputs in the object
                self.best_fit_index = best_fit_index
                self.best_fit = best_fit
                self.model_flux = model_flux
                self.dist_lum = dist_lum
                self.age = np.array(age)
                self.metal = np.array(metal)
                self.mass_weights = mass_weights
                self.light_weights = light_weights
                self.chis = chis
                self.branch = branch
                self.unnorm_mass = unnorm_mass
                self.probs = probs
                self.best_fit = best_fit
                self.averages = averages
                self.wave = wave

                bf_mass = (self.mass_weights[self.best_fit_index]>0)[0]
                bf_light = (self.light_weights[self.best_fit_index]>0)[0]
                mass_per_ssp = self.unnorm_mass[self.best_fit_index[0]][bf_mass]*self.flux_units* 4 * np.pi * self.dist_lum**2.0

                age_per_ssp = self.age[bf_mass]
                metal_per_ssp = self.metal[bf_mass]
                weight_mass_per_ssp = self.mass_weights[self.best_fit_index[0]][bf_mass]
                weight_light_per_ssp = self.light_weights[self.best_fit_index[0]][bf_light]
                order = np.argsort(-weight_light_per_ssp)

                # Do we want to put this all in another function??
                # We could provide it with the arrays and call something like get_massloss_parameters()?
                # I think it looks a little untidy still because of my bad coding.
    
                # Gets the mass loss factors.
                imf_name = dict_imfs[imf].lower()
                ML_metallicity, ML_age, ML_totM, ML_alive, ML_wd, ML_ns, ML_bh, ML_turnoff = np.loadtxt(
                        os.path.join(os.environ['STELLARPOPMODELS_DIR'],'data','massloss_'+imf_name+'.txt'), 
                        unpack=True, skiprows=2)

                # First build the grids of the quantities. Make sure they are in linear units.                  
                estimate_ML_totM    = estimation(10**ML_metallicity, ML_age, ML_totM)
                estimate_ML_alive   = estimation(10**ML_metallicity, ML_age, ML_alive)
                estimate_ML_wd      = estimation(10**ML_metallicity, ML_age, ML_wd)
                        
                estimate_ML_ns      = estimation(10**ML_metallicity, ML_age, ML_ns) 
                estimate_ML_bh      = estimation(10**ML_metallicity, ML_age, ML_bh) 
                estimate_ML_turnoff = estimation(10**ML_metallicity, ML_age, ML_turnoff)

                # Now loop through SSPs to find the nearest values for each.
                final_ML_totM, final_ML_alive, final_ML_wd, final_ML_ns, final_ML_bh, final_ML_turnoff, final_gas_fraction = \
                        [], [], [], [], [], [], []
                for i_ssp in range(len(age_per_ssp)):
                    new_ML_totM = estimate_ML_totM.estimate(metal_per_ssp[i_ssp], age_per_ssp[i_ssp])
                    new_ML_alive = estimate_ML_alive.estimate(metal_per_ssp[i_ssp], age_per_ssp[i_ssp])
                    new_ML_wd = estimate_ML_wd.estimate(metal_per_ssp[i_ssp], age_per_ssp[i_ssp])
                    new_ML_ns = estimate_ML_ns.estimate(metal_per_ssp[i_ssp], age_per_ssp[i_ssp])
                    new_ML_bh = estimate_ML_bh.estimate(metal_per_ssp[i_ssp], age_per_ssp[i_ssp])
                    new_ML_turnoff = estimate_ML_turnoff.estimate(metal_per_ssp[i_ssp], age_per_ssp[i_ssp])

                    final_ML_totM.append(mass_per_ssp[i_ssp] * new_ML_totM)
                    final_ML_alive.append(mass_per_ssp[i_ssp] * new_ML_alive)
                    final_ML_wd.append(mass_per_ssp[i_ssp] * new_ML_wd)
                    final_ML_ns.append(mass_per_ssp[i_ssp] * new_ML_ns)
                    final_ML_bh.append(mass_per_ssp[i_ssp] * new_ML_bh)
                    final_ML_turnoff.append(mass_per_ssp[i_ssp] * new_ML_turnoff)
                    final_gas_fraction.append(mass_per_ssp[i_ssp] - new_ML_totM)
                
        	final_ML_totM = np.array(final_ML_totM) 
                final_ML_alive = np.array(final_ML_alive)
                final_ML_wd = np.array(final_ML_wd)
                final_ML_ns = np.array(final_ML_ns)  
                final_ML_bh = np.array(final_ML_bh) 
                final_ML_turnoff = np.array(final_ML_turnoff)
                final_gas_fraction = np.array(final_gas_fraction)

                # Calculate the total mass loss from all the SSP contributions.
                combined_ML_totM        = np.sum(final_ML_totM)
                combined_ML_alive       = np.sum(final_ML_alive)
                combined_ML_wd          = np.sum(final_ML_wd)
                combined_ML_ns          = np.sum(final_ML_ns)
                combined_ML_bh          = np.sum(final_ML_bh)
                combined_gas_fraction   = np.sum(mass_per_ssp - final_ML_totM)

                # 8. output dictionary 
                output = {} 
                output['wavelength'] = wave # Angstrom
                output['flux_data'] = data_flux # 1e-17erg/s/cm2/Angstrom
                output['flux_error'] = error_flux # 1e-17erg/s/cm2/Angstrom
                output['flux_bestfit'] = best_fit # 1e-17erg/s/cm2/Angstrom
                output['properties'] = {} 
                output['properties']['redshift'] = self.specObs.redshift
                output['properties']['IMF'] = dict_imfs[imf]
                output['properties']['Model'] = mod

                for prop in ['age', 'metallicity']: 
                    for lm in ['light', 'mass']: 
                        if prop == 'metallicity': propstr = 'metal'
                        else: propstr = prop
                        output['properties'][prop+'_'+lm+'W'] = trylog10(averages[lm+'_'+propstr])
                        for isig in [1, 2, 3]: 
                            output['properties'][prop+'_'+lm+'W_up_'+str(isig)+'sig'] = \
                                    trylog10(averages[lm+'_'+propstr+'_'+str(isig)+'_sig_plus'])
                            output['properties'][prop+'_'+lm+'W_low_'+str(isig)+'sig'] = \
                                    trylog10(averages[lm+'_'+propstr+'_'+str(isig)+'_sig_minus'])

                output['properties']['total_mass'] = trylog10(averages['stellar_mass'])
                output['properties']['stellar_mass'] = trylog10(combined_ML_alive+combined_ML_wd+combined_ML_ns+combined_ML_bh)
                output['properties']['living_stars_mass'] = trylog10(combined_ML_alive)
                output['properties']['remnant_mass'] = trylog10(combined_ML_wd+combined_ML_ns+combined_ML_bh)
                output['properties']['remnant_mass_in_whitedwarfs'] = trylog10(combined_ML_wd)
                output['properties']['remnant_mass_in_neutronstars'] = trylog10(combined_ML_ns)
                output['properties']['remnant_mass_blackholes'] = trylog10(combined_ML_bh)
                output['properties']['mass_of_ejecta'] = trylog10(combined_gas_fraction)
                output['properties']['total_mass_up_1sig'] = trylog10(averages['stellar_mass_1_sig_plus'])
                output['properties']['total_mass_low_1sig'] = trylog10(averages['stellar_mass_1_sig_minus'])
                output['properties']['total_mass_up_2sig'] = trylog10(averages['stellar_mass_2_sig_plus'])
                output['properties']['total_mass_low_2sig'] = trylog10(averages['stellar_mass_2_sig_minus'])
                output['properties']['total_mass_up_3sig'] = trylog10(averages['stellar_mass_3_sig_plus'])
                output['properties']['total_mass_low_3sig'] = trylog10(averages['stellar_mass_3_sig_minus'])
                output['properties']['EBV'] = best_ebv
                output['properties']['ssp_i_ssp'] =len(order)

                # quantities per SSP
                for iii in range(len(order)):
                    output['properties']['total_mass_ssp_'+str(iii)] = trylog10(mass_per_ssp[order][iii])
                    output['properties']['stellar_mass_ssp_'+str(iii)] = trylog10(final_ML_alive[order][iii]+final_ML_wd[order][iii]+final_ML_ns[order][iii]+final_ML_bh[order][iii])
                    output['properties']['living_stars_mass_ssp_'+str(iii)] = trylog10(final_ML_alive[order][iii])	
                    output['properties']['remnant_mass_ssp_'+str(iii)] = trylog10(final_ML_wd[order][iii]+final_ML_ns[order][iii]+final_ML_bh[order][iii])
                    output['properties']['remnant_mass_in_whitedwarfs_ssp_'+str(iii)] = trylog10(final_ML_wd[order][iii])
                    output['properties']['remnant_mass_in_neutronstars_ssp_'+str(iii)] = trylog10(final_ML_ns[order][iii])
                    output['properties']['remnant_mass_in_blackholes_ssp_'+str(iii)] = trylog10(final_ML_bh[order][iii])
                    output['properties']['mass_of_ejecta_ssp_'+str(iii)] = trylog10(mass_per_ssp[order][iii] - final_ML_totM[order][iii])
                    output['properties']['log_age_ssp_'+str(iii)] = trylog10(age_per_ssp[order][iii])
                    output['properties']['metal_ssp_'+str(iii)] = trylog10(metal_per_ssp[order][iii])
                    output['properties']['SFR_ssp_'+str(iii)] = trylog10(mass_per_ssp[order][iii]/age_per_ssp[order][iii])	
                    output['properties']['weightMass_ssp_'+str(iii)] = weight_mass_per_ssp[order][iii]
                    output['properties']['weightLight_ssp_'+str(iii)] = weight_light_per_ssp[order][iii]

            output['properties']['model'] = self.models
            output['properties']['ageMin'] = self.age_limits[0]
            output['properties']['ageMax'] = self.age_limits[1]
            output['properties']['Zmin'] = self.Z_limits[0]
            output['properties']['Zmax'] = self.Z_limits[1]

            if self.write_results: # write to hdf5 file 
                f = h5py.File(self.outputFile, 'w') 
                for k in output.keys(): 
                    if k == 'properties': 
                        grp = f.create_group('properties')
                        for kk in output['properties'].keys(): 
                            grp.create_dataset(kk, data=output['properties'][kk])
                    else: 
                        f.create_dataset(k, data=output[k]) 
                f.close() 
            return output 

    def Fitter(self, wavelength, data, error, models): 
        ''' wrapper for newFitter
        '''
        nfitter = newFitter(wavelength, data, error, models, upper_limit_fit=self.max_iterations, fit_cap=self.fit_per_iteration_cap) 
        return nfitter.output() 
