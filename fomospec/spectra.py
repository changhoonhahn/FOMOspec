'''
'''
import numpy as np 
from astropy import units as U
# --- desi ---
import desispec.io as desiIO
# --- firefly ---
import GalaxySpectrumFIREFLY as gs


class GalaxySpec(object): 
    ''' General wrapper class to interace with the different spectral 
    fitting code.
    '''
    def __init__(self): 
        pass


class GSfirefly(GalaxySpec): 
    ''' child class of firefly spectral fitting code
    '''
    def __init__(self): 
        self.milky_way_reddening = True 
        self.hpf_mode = 'on' 
        self.N_angstrom_masked = 20. 

        # wavelength of masking emission lines
        self.lambda_emission = [3728., 4861., 5007., 6564.]

    def generic(wave, flux, fractional_error=0.1):  
        ''' Read in given restframe wavelength [Ang] and flux [1e-17 erg/s/A/cm^2] 
        to class compatible with gs.GalaxySpectrumFIREFLY object
        '''
        self.ra = 0.
        self.dec = 0.
        self.redshift = 0.01
		
        area = 4. * np.pi * cosmo.luminosity_distance(self.redshift).to(U.cm)**2.
        self.restframe_wavelength = wave
        self.wavelength = self.restframe_wavelength * (1. + self.redshift)
        
        self.flux = flux
        self.error = self.flux * fractional_error
        self.bad_flags = np.ones(len(self.restframe_wavelength))
		
        vdisp = 70.
        trust_flag = 1
        objid = 0

        lines_mask = self.emissionlineMask(self.restframe_wavelength)

        self.wavelength = self.wavelength[~lines_mask]
        self.restframe_wavelength = self.restframe_wavelength[~lines_mask]

        influx = self.flux[~lines_mask] 
        inerror = self.error[~lines_mask] 
        inbad_flags = self.bad_flags[~lines_mask] 		

        self.flux, self.error, self.bad_flags = gs.remove_bad_data(influx, inerror, inbad_flags)

        self.r_instrument = np.zeros(len(self.wavelength))
        for wi, w in enumerate(self.wavelength):
            if w < 6000:
                self.r_instrument[wi] = (2270.0-1560.0)/(6000.0-3700.0)*w + 420.0 
            else:
                self.r_instrument[wi] = (2650.0-1850.0)/(9000.0-6000.0)*w + 250.0 
        self.ebv_mw = 0.0
        return None 

    def emissionlineMask(self, restframe_wavelength): 
        ''' given restframe wavelength return mask where emission lines are 
        '''
        lines_mask = np.ones(len(restframe_wavelength)).astype(bool)
        for lam in self.lambda_emission: 
            lines_mask = lines_mask | ((restframe_wavelength > lam - self.N_angstrom_masked) & 
                    (restframe_wavelength < lam + self.N_angstrom_masked))
        return lines_mask 
    

"""
    class galSpecFIREFLY(gs.GalaxySpectrumFIREFLY): 
        ''' child class of FIREFLY class `GalaxySpectrumFIREFLY` which is 
        a wrapper for reading in spectra from different sources.  
        '''

        def openDESIsynspec(self): 
            # read in desi-like synthetic spectra
            spec_desi = desiIO.read_spectra(self.path_to_spectrum) 

            self.ra = None
            self.dec = None 
            
            self.wavelength = np.concatenate([spec_desi.wave['b'], spec_desi.wave['r'],  spec_desi.wave['z']]) 
            self.flux = np.concatenate([spec_desi.band['b'][0], spec_desi.band['r'][0], spec_desi.band['z'][0]]) 
            self.error = np.concatenate([spec_desi.ivar['b'][0], spec_desi.ivar['r'][0], spec_desi.ivar['z'][0]])**(-0.5)
            self.bad_flags = np.ones(len(self.wavelength))
            self.redshift = np.zeros(len(self.wavelength))
            self.vdisp = np.zeros(len(self.wavelength))
            self.restframe_wavelength = self.wavelength / (1.0+self.redshift)

            self.trust_flag = 1
            self.objid = 0

            # masking emission lines
            lambda_emission = [3728., 4861., 5007., 6564.]
            lines_mask = np.ones(len(self.restframe_wavelength)).astype(bool)
            for lam in lambda_emission: 
                lines_mask = lines_mask | ((self.restframe_wavelength > lam - self.N_angstrom_masked) & (self.restframe_wavelength < lam + self.N_angstrom_masked))
            self.wavelength = self.wavelength[~lines_mask]
            self.restframe_wavelength = self.restframe_wavelength[~lines_mask]
            
            influx = self.flux[~lines_mask]
            inerror = self.error[~lines_mask]

            inbad_flags = self.bad_flags[~lines_mask]
            self.flux, self.error, self.bad_flags = gs.remove_bad_data(influx, inerror, inbad_flags)
            self.r_instrument = np.zeros(len(self.wavelength))

            for wi,w in enumerate(self.wavelength):
                    if w<6000:
                            self.r_instrument[wi] = (2270.0-1560.0)/(6000.0-3700.0)*w + 420.0
                    else:
                            self.r_instrument[wi] = (2650.0-1850.0)/(9000.0-6000.0)*w + 250.0

            self.milky_way_reddening = False
            self.ebv_mw = 0.0
"""
