#!/bin/usr/python
import sys 
import numpy as np
import corner as DFM
from astropy.io import fits
# -- desi -- 
import desispec.io as desiIO
# --- fomospec ---
from FOMOspec import util as UT
from FOMOspec import fitters as Fitters

import matplotlib.pyplot as plt

def prospector_mockdata(infer_method='dynesty'): 
    '''
    '''
    prosp = Fitters.Prospector()

    # mock data 
    tt = np.array([-0.5, 0.3, 3., 12., 1e10])
    zred = 0.1 
    mock_lam = np.linspace(3000, 1e4, 1e3)
    mock_flux_maggies, _, _ = prosp.model(mock_lam, tt, zred)
    # convert to 10^-17 ergs/s/cm^2/Ang
    mock_flux = mock_flux_maggies * 1e17 * UT.c_light() / mock_lam**2 * (3631. * UT.jansky_cgs())
    
    if infer_method == 'dynesty': 
        # run dynamic nested sampling 
        prosp.dynesty_spec(mock_lam, mock_flux, None, zred, 
                nested=True, maxcall_init=50000, maxcall=50000, 
                write=True, output_file=''.join([UT.dat_dir(), 'prospector_dynesty_mock_test.h5']))
    elif infer_method == 'emcee': 
        # emcee 
        prosp.emcee_spec(mock_lam, mock_flux, None, zred, 
                write=True, output_file=''.join([UT.dat_dir(), 'prospector_emcee_mock_test.h5']),
                silent=False)
    return None 


def prospector_LGAL_sourceSpec_i(galid, infer_method='dynesty'): 
    ''' run prospector on L-Gal source spectra 
    '''
    # read in source spectra
    f_name = 'gal_spectrum_'+str(galid)+'_BGS_template_BC03_Stelib.fits'
    f_inspec = fits.open(''.join([UT.dat_dir(), 'Lgal/templates/', f_name]))
    specin = f_inspec[1].data

    zred = f_inspec[0].header['REDSHIFT']
    wave = specin['wave']
    flux = specin['flux_dust_nonoise'] * 1e-4 * 1e7 # in units of erg/s/A/cm2
    # convert to maggies 
    flux = flux / UT.c_light() * wave**2 / (3631. *  UT.jansky_cgs()) # maggies 

    prosp = Fitters.Prospector()
    if infer_method == 'dynesty': 
        # run dynamic nested sampling 
        f_name = ''.join(['prospector.dynesty.gal_spectrum_', str(galid), '_BGS_template_BC03_Stelib.h5']) 
        prosp.dynesty_spec(wave, flux, None, zred, 
                nested=True, maxcall_init=50000, maxcall=100000, 
                write=True, output_file=f_name) 
    elif infer_method == 'emcee': 
        # emcee 
        prosp.emcee_spec(mock_lam, mock_flux, None, zred, 
                write=True, output_file=''.join([UT.dat_dir(), 'prospector_emcee_mock_test.h5']))
    return None 


def prospector_LGAL_desiSpec_i(galid, mask=False, infer_method='dynesty'): 
    ''' run prospector on L-Gal DESI-like spectra
    '''
    # read in source spectra (only to get the redshift) 
    f_name = 'gal_spectrum_'+str(galid)+'_BGS_template_BC03_Stelib.fits'
    f_inspec = fits.open(''.join([UT.dat_dir(), 'Lgal/templates/', f_name]))
    zred = f_inspec[0].header['REDSHIFT']

    # read desi-like spectra
    f_name = 'gal_spectrum_'+str(galid)+'_BGS_template_BC03_Stelib.fits'
    f_outspec = ''.join([UT.dat_dir(), 'Lgal/spectra/', 'desi_out_', f_name]) 
    spec_desi = desiIO.read_spectra(f_outspec)

    wave = np.concatenate([spec_desi.wave[b] for b in ['b', 'r', 'z']]) 
    flux = np.concatenate([spec_desi.flux[b][0] for b in ['b', 'r', 'z']]) # 10-17 ergs/s/cm2/AA
    flux_unc = np.concatenate([spec_desi.ivar[b][0]**-0.5 for b in ['b', 'r', 'z']]) 
    #flux *= 1e-17 / UT.c_light() * wave**2 / (3631. *  UT.jansky_cgs()) # maggies 
    #flux_unc *= 1e-17 / UT.c_light() * wave**2 / (3631. *  UT.jansky_cgs()) # maggies 
    #plt.errorbar(wave, flux, flux_unc, fmt='.k')
    #plt.ylim([0., flux.max()]) 
    #plt.show() 
    #raise ValueError
    
    prosp = Fitters.Prospector()
    if infer_method == 'dynesty': 
        # run dynamic nested sampling 
        if mask:
            f_name = ''.join(['prospector.dynesty.masked.gal_spectrum_', str(galid), '_BGS_template_BC03_Stelib.h5']) 
        else: 
            f_name = ''.join(['prospector.dynesty.gal_spectrum_', str(galid), '_BGS_template_BC03_Stelib.h5']) 
        prosp.dynesty_spec(wave, flux, flux_unc, zred, mask=mask,
                nested=True, maxcall_init=25000, maxcall=50000, 
                write=True, output_file=f_name) 
    elif infer_method == 'emcee': 
        # emcee 
        if mask:
            f_name = ''.join([UT.dat_dir(), 
                'prospector.emcee.masked.gal_spectrum_', str(galid), '_BGS_template_BC03_Stelib.h5']) 
        else: 
            f_name = ''.join([UT.dat_dir(), 
                'prospector.emcee.gal_spectrum_', str(galid), '_BGS_template_BC03_Stelib.h5']) 
        prosp.emcee_spec(wave, flux, flux_unc, zred, mask=mask, 
                write=True, output_file=f_name, silent=False)
    return None 
    

if __name__=="__main__": 
    type = sys.argv[1] 
    infer = sys.argv[2]

    if type == 'source': 
        galid = sys.argv[3] 
        prospector_LGAL_sourceSpec_i(galid, infer_method=infer)
    elif type == 'desi': 
        galid = sys.argv[3] 
        imask = int(sys.argv[4]) 
        if imask == 0: mask = False
        elif imask == 1: mask = True
        else: raise ValueError
        prospector_LGAL_desiSpec_i(galid, mask=mask, infer_method=infer)
    elif type == 'mock': 
        prospector_mockdata(infer_method=infer)
