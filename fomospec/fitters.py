import numpy as np 
from astropy import units as U
from astropy.cosmology import Planck13 as cosmo

from estimations_3d import estimation
# --- firefly ---
import GalaxySpectrumFIREFLY as gs
import StellarPopulationModel as spm
from StellarPopulationModel import trylog10
from firefly_library import airtovac, convert_chis_to_probs, light_weights_to_mass, calculate_averages_pdf, normalise_spec, match_data_models
from firefly_dust import hpf, unred, determine_attenuation
from firefly_instrument import downgrade
from firefly_fitter import fitter


dict_imfs = {'cha': 'Chabrier', 'ss': 'Salpeter', 'kr': 'Kroupa'}


class Firefly(spm.StellarPopulationModel): 
    def fit_models_to_data(self): 
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
	        # print "getting the models"
                model_wave_int, model_flux_int, age, metal = self.get_model(
                        mod, # model 
                        imf, # IMF 
                        self.deltal_libs[mi],
                        self.specObs.vdisp, 
                        self.specObs.restframe_wavelength, 
                        self.specObs.r_instrument, 
                        self.specObs.ebv_mw)
                # B. matches the model and data to the same resolution
                # print "Matching models to data"
                wave, data_flux, error_flux, model_flux_raw = match_data_models(
                        self.specObs.restframe_wavelength, 
                        self.specObs.flux, 
                        self.specObs.bad_flags, 
                        self.specObs.error, 
                        model_wave_int, model_flux_int, self.wave_limits[0], self.wave_limits[1], saveDowngradedModel=False)
                # C. normalises the models to the median value of the data
                # print "Normalising the models"
                model_flux, mass_factors = normalise_spec(data_flux, model_flux_raw)

                # 3. Corrects from dust attenuation
                if self.hpf_mode=='on':
                    # 3.1. Determining attenuation curve through HPF fitting, apply 
                    # attenuation curve to models and renormalise spectra
                    best_ebv, attenuation_curve = determine_attenuation(wave, data_flux, error_flux, model_flux, self, age, metal)
                    model_flux_atten = np.zeros(np.shape(model_flux_raw))
                    for m in range(len(model_flux_raw)):
                        model_flux_atten[m] = attenuation_curve * model_flux_raw[m] 
                        
                        model_flux, mass_factors = normalise_spec(data_flux, model_flux_atten)
                        # 4. Fits the models to the data
                        light_weights, chis, branch = fitter(wave, data_flux, error_flux, model_flux, self)

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
                    hpf_models,mass_factors = normalise_spec(hpf_data,hpf_models)
                    # 4. Fits the models to the data
                    light_weights, chis, branch = fitter(wave, hpf_data,hpf_error, hpf_models, self)

                # 5. Get mass-weighted SSP contributions using saved M/L ratio.
                unnorm_mass, mass_weights = light_weights_to_mass(light_weights, mass_factors)

                if np.all(np.isnan(mass_weights)):
                    raise ValueError # tbhdu = self.create_dummy_hdu()

                # print "Calculating average properties and outputting"
                # 6. Convert chis into probabilities and calculates all average properties and errors
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
                        os.path.join(os.environ['STELLARPOPMODELS_DIR'],'data','massloss_', imf_name, '.txt'), 
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

                # Calculate the total mass loss from all the SSP contributions.
                combined_ML_totM        = np.sum(np.array(final_ML_totM))
                combined_ML_alive       = np.sum(np.array(final_ML_alive))
                combined_ML_wd          = np.sum(np.array(final_ML_wd))
                combined_ML_ns          = np.sum(np.array(final_ML_ns))
                combined_ML_bh          = np.sum(np.array(final_ML_bh))
                combined_gas_fraction   = np.sum(mass_per_ssp - np.array(final_ML_totM))

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
                        output['properties'][prop+'_'+lm+'W'] = trylog10(averages[lm+'_'+prop])
                        for isig in [1, 2, 3]: 
                            output['properties'][prop+'_'+lm+'W_up_'+str(isig)+'sig'] = \
                                    trylog10(averages[lm+'_'+prop+'_'+str(isig)+'_sig_plus'])
                            output['properties'][prop+'_'+lm+'W_low_'+str(isig)+'sig'] = \
                                    trylog10(averages[lm+'_'+prop+'_'+str(isig)+'_sig_minus'])

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
            return output 
