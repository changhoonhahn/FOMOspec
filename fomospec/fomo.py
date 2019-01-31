'''

forward modeling galaxy spectra from star formation and 
chemical enrichment histories


'''
import fsps 
import numpy as np 
from astropy import units as U 
from astropy.cosmology import Planck13
# -- fomospec -- 
from . import util as UT 


def FSPS_nodust(tage, sfh, zh, imf='chabrier'):
    ''' take star formation and chemical histories and get 
    the composite stellar population spectra. This is a simple
    wrapper for python-fsps 

    :param tage:    
        age of stellar population; also look back time not to be
        confused with cosmic time. In units of Gyr!

    :param sfh:     
        star formation history -- i.e. stellar mass formed
        by the SSP at tage and Zh(tage). This is what the SSP
        will be normalized by. In units of Msun (not log Msun) 

    :param zh:      
        metallicity history. Note this is Z not log Z

    :param imf: (default: chabrier) 
        imf assumed by the ssp 
    
    :returns wave_rest:
        rest frame wavlength of SSPs in angstroms

    :returns lum_ssp: 
        luminosity of the SSPs [n_ssp, n_wavelength] in units 
        of Lsun/AA
    '''
    i_imf = {'chabrier': 1}  # expand this dictionary as we try other stuff

    ssp = fsps.StellarPopulation(
            zcontinuous=1,          # interpolate metallicities
            sfh=0,                  # returns SSP
            imf_type=i_imf[imf])    # default is chabrier 

    for i, t, m, z in zip(range(len(tage)), tage, sfh, zh): 
        ssp.params['logzsol'] = np.log10(z/0.0190) # log(Z/Zsun) 
        wave_rest, lum_i = ssp.get_spectrum(tage=t, peraa=True) # in units of Lsun/AA

        if i == 0: lum_ssp = np.zeros((len(tage), len(wave_rest)))
        if m > 0.: 
            lum_ssp[i,:] = m * lum_i # Mformed * SSP 
    return wave_rest, lum_ssp 


def zSpectrum(w_rest, lum_ssp, zred, cosmo=Planck13): 
    ''' redshift the SSP luminosity [Lsun/AA] and return 
    spectrum flux [erg/s/cm^2/A]

    :params w_rest: 
        rest-frame wavelength

    :params lum_ssp: 
        luminosity of SSPs in units of Lsun/AA (FSPS output) 

    :params zred: 
        redshift 

    :params cosmo: (default astropy.cosmology.Planck13) 
        cosmology 

    :returns w_obs: 
        observed-frame wavelength

    :returns flux: 
        redshifted flux of SSPs derived from lum_ssp.
        In units of erg/s/cm^2/Angstrom
    '''
    d_lum = cosmo.luminosity_distance(zred).to(U.cm).value # luminosity distance in cm

    w_obs = w_rest * (1. + zred) # observed-frame wavelength 
    flux = lum_ssp * UT.Lsun() / (4. * np.pi * d_lum**2) / (1. + zred) 
    return w_obs, flux
