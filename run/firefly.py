#!/bin/usr/python
import sys 
import numpy as np 
from astropy.io import fits
import astropy.cosmology as co
# -- desi -- 
import desispec.io as desiIO
# -- FOMOspec -- 
from fomospec import util as UT
from fomospec import spectra as Spec
from fomospec import fitters as Fitters


def firefly_LGAL_sourceSpec(galid, model='m11', model_lib='MILES', imf='cha', hpf_mode='on'): 
    ''' run firefly on L-Gal source spectra 
    '''
    # read in source spectra
    f_name = 'gal_spectrum_'+str(galid)+'_BGS_template_BC03_Stelib.fits'
    f_inspec = fits.open(''.join([UT.dat_dir(), 'Lgal/templates/', f_name]))
    specin = f_inspec[1].data

    spec_in = {}
    spec_in['redshift'] = f_inspec[0].header['REDSHIFT']
    spec_in['wave'] = specin['wave']
    spec_in['flux'] = specin['flux_dust_nonoise'] * 1e-4 * 1e7 *1e17 #from W/A/m2 to 10e-17 erg/s/A/cm2
    spec_in['flux_dust_nonoise'] = specin['flux_dust_nonoise'] * 1e-4 * 1e7 *1e17 
    spec_in['flux_nodust_nonoise'] = specin['flux_nodust_nonoise'] * 1e-4 * 1e7 *1e17
    
    # structure source spectra for firefly read in 
    gspec = Spec.GSfirefly()
    gspec.generic(spec_in['wave'], spec_in['flux_dust_nonoise'], redshift=spec_in['redshift'])
    gspec.path_to_spectrum = UT.dat_dir()
    
    # output firefly file 
    f_name = ''.join(['firefly.', model, '.', model_lib, '.imf_', imf, '.dust_', hpf_mode, '.', 
        'gal_spectrum_', str(galid), '_BGS_template_BC03_Stelib.hdf5']) 
    f_firefly = ''.join([UT.dat_dir(), 'Lgal/templates/', f_name])

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


def firefly_LGAL_desiSpec(galid, model='m11', model_lib='MILES', imf='cha', hpf_mode='on'): 
    ''' run firelfy on L-Gal DESI-like spectra
    '''
    # read in source spectra (only to get the redshift) 
    f_name = 'gal_spectrum_'+str(galid)+'_BGS_template_BC03_Stelib.fits'
    f_inspec = fits.open(''.join([UT.dat_dir(), 'Lgal/templates/', f_name]))
    redshift = f_inspec[0].header['REDSHIFT']

    # read desi-like spectra
    f_name = 'gal_spectrum_'+str(galid)+'_BGS_template_BC03_Stelib.fits'
    f_outspec = ''.join([UT.dat_dir(), 'Lgal/spectra/', 'desi_out_', f_name]) 
    spec_desi = desiIO.read_spectra(f_outspec)
    
    gspec = Spec.GSfirefly()
    gspec.DESIlike(spec_desi, redshift=redshift)
    gspec.path_to_spectrum = UT.dat_dir()

    # output firefly file 
    f_out = ''.join(['firefly.', model, '.', model_lib, '.imf_', imf, '.dust_', hpf_mode, '.', 
        'desi_out_', f_name.split('.fits')[0], '.hdf5']) 
    f_firefly = ''.join([UT.dat_dir(), 'Lgal/spectra/', f_out])

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


if __name__=='__main__':
    galid = int(sys.argv[1])
    type = sys.argv[2]
    dust = sys.argv[3]
    if type == 'source': 
        firefly_LGAL_sourceSpec(galid, hpf_mode=dust) 
    elif type == 'desi': 
        firefly_LGAL_desiSpec(galid, hpf_mode=dust)
