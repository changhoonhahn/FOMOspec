'''

run spectral fitter through very simple test 


'''
import h5py
import fsps
import numpy as np
from astropy.cosmology import FlatLambdaCDM
# --- fomospec ---
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


age_bounds = np.array(((0.000,0.0025,0.005),(0.005,0.010,0.015),(0.015,0.020,0.025),(0.025,0.030,0.035),(0.035,0.0400,0.045),(0.045,0.050,0.055),(0.055,0.060,0.065),(0.065,0.070,0.075),(0.075,0.0800,0.085),(0.085,0.090,0.095),(0.095,0.100,0.125),(0.125,0.150,0.175),(0.175,0.20,0.225),(0.225,0.250,0.275),(0.275,0.300,0.325),(0.325,0.350,0.375),(0.375,0.40,0.425),(0.425,0.450,0.475),(0.475,0.5125,0.55),(0.550,0.600,0.650),(0.650,0.70,0.750),(0.75,0.80,0.85),(0.85,0.90,0.95),(0.95,1.0375,1.125),(1.125,1.25,1.375),(1.375,1.50,1.625),(1.625,1.75,1.875),(1.875,2.00,2.125),(2.125,2.25,2.375),(2.375,2.50,2.625),(2.625,2.75,2.875),(2.875,3.00,3.125),(3.125,3.25,3.375),(3.375,3.50,3.625),(3.625,3.75,3.875),(3.875,4.00,4.25),(4.25,4.50,4.75),(4.75,5.00,5.25),(5.25,5.50,5.75),(5.75,6.00,6.25),(6.25,6.50,6.75),(6.75,7.00,7.25),(7.25,7.50,7.75),(7.75,8.00,8.25),(8.25,8.5,8.75),(8.75,9.0,9.25),(9.25,9.5,9.75),(9.75,10.0,10.25),(10.25,10.5,10.75),(10.75,11.0,11.25),(11.25,11.5,11.75),(11.75,12.0,12.25),(12.25,12.5,12.75),(12.75,13.0,13.25),(13.25,13.5,13.75),(13.75,13.875,14.0)))

Z_mid = [-2.5, -2.05, -1.75, -1.45, -1.15, -0.85, -0.55, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.4, 0.5]


def simple_spectra(igal, i_metal=5, zred=0.01):
    ''' construct simple spectra with single metallicity, no dust, no nonsense  
    '''
    cosmo = FlatLambdaCDM(70., 0.3) 

    # read in SFH and M_* 
    f_eagle = h5py.File(''.join([UT.dat_dir(), '0EAGLE_SFRHs.hdf5']), 'r') 
    sfh = f_eagle['SFRH'].value 
    mtot = f_eagle['TotalMassFormed'].value  

    sp = fsps.StellarPopulation(zcontinuous=1, sfh=3, tage=cosmo.lookback_time(zred).value)
    sfh = np.zeros((56,2))
    #sfh[:,0] = 14-age_bounds[:,1][::-1]
    sfh[:,0] = age_bounds[:,1]
    sfh[:,1] = f_eagle['SFRH'][igal][:,i_metal]
    sp.set_tabular_sfh(sfh[:,0],sfh[:,1]) # feed in tabulated SFH
    sp.params['logzsol'] = Z_mid[i_metal] #
    
    wave, spec = sp.get_spectrum(peraa=False)  # Lsun/Hz
    # convert flux to ergs/s/cm^2/AA
    d_lum = cosmo.luminosity_distance(zred).value * 1e6 * UT.parsec() # luminosity distance (pc) 
    spec *= UT.Lsun() * UT.c_light() / wave**2 / (4 * np.pi * d_lum**2)
    print np.log10(np.sum(sp.stellar_mass))
    
    # save spectra 
    f_spec = ''.join([UT.dat_dir(), 
            'simple_spectra.', str(igal), '.metal', str(i_metal), '.z', str(zred), '.dat'])
    np.savetxt(f_spec, np.vstack([wave, np.sum(spec, axis=0)]).T)

    fig = plt.figure(figsize=(8,4))
    sub = fig.add_subplot(111)
    sub.plot(wave, np.sum(spec, axis=0), c='k', lw=1)
    sub.set_xlabel('wavelength [$A$]', fontsize=25) 
    sub.set_xlim([3500., 1e4]) 
    sub.set_ylabel('flux [$ergs/s/cm^2/A$]', fontsize=20) 
    fig.savefig(''.join([UT.fig_dir(), 'simple_spectra.', str(igal), '.metal', str(i_metal), '.z', str(zred), '.png']), 
    bbox_inches='tight') 
    return None 


def simple_firefly(igal, i_metal=5, zred=0.01):
    '''
    '''
    cosmo = FlatLambdaCDM(70., 0.3) 
    f_spec = ''.join([UT.dat_dir(), 
            'simple_spectra.', str(igal), '.metal', str(i_metal), '.z', str(zred), '.dat'])
    wave, spec = np.loadtxt(f_spec, unpack=True, usecols=[0,1]) 

    gspec = Spec.GSfirefly()
    gspec.generic(wave*(1.+zred), spec*1e17, redshift=zred)
    gspec.path_to_spectrum = UT.dat_dir()
    f_firefly = ''.join([UT.dat_dir(), 'firefly.', f_spec.rsplit('/', 1)[1].rsplit('.dat', 1)[0], '.hdf5']) 
    if i_metal == 5: 
        Zlim = [10**(Z_mid[i_metal]-0.1), 10**(Z_mid[i_metal]+0.1)]
    elif i_metal == 0: 
        Zlim = [1e-4, 0.005] #000345349769783

    firefly = Fitters.Firefly(gspec,
            f_firefly,          # output file 
            cosmo,              # comsology
            models = 'm11',     # model ('m11', 'bc03', 'm09') 
            model_libs = ['MILES'], # model library for M11
            imfs = ['kr'],      # IMF used ('ss', 'kr', 'cha')
            hpf_mode = 'on', # uses HPF to dereden the spectrum                       
            age_limits = [0., 10**(14-9)], 
            wave_limits = [3350., 9000.], 
            suffix=None, 
            downgrade_models = False, 
            data_wave_medium = 'vacuum', 
            Z_limits = Zlim, 
            use_downgraded_models = True, 
            write_results = True)
    bestfit = firefly.fit_models_to_data(silent=True)
    print firefly.best_fit_index
    print firefly.age#, 
    print firefly.metal#, firefly.mass_weights, firefly.light_weights, firefly.chis 

    # read in firefly fit 
    ffly_out, ffly_prop = UT.readFirefly(f_firefly) 
    mmed = 10**ffly_prop['total_mass']

    # spectra comparison
    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(111)
    sub.plot(wave, spec*1e17, c='C0', lw=1, label='source')
    sub.plot(ffly_out['wavelength'], ffly_out['flux_bestfit'], c='C1', ls='--', label='FIREFLY best-fit')
    sub.legend(loc='upper right', fontsize=20)
    sub.set_xlabel('Rest-frame Wavelength [$\AA$]', fontsize=25)
    sub.set_xlim([3.3e3, 9.8e3])
    sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/\AA$]', fontsize=25)
    sub.set_ylim([0., 1.25e17*spec[(wave > 3e3) & (wave < 1e4)].max()])
    fig.savefig(''.join([UT.fig_dir(), 'simple_firefly_spectra.', str(igal), '.metal', str(i_metal), '.z', str(zred), '.png']), 
        bbox_inches='tight') 

    # comparison of SFH 
    f_eagle = h5py.File(''.join([UT.dat_dir(), '0EAGLE_SFRHs.hdf5']), 'r') 
    sfh = f_eagle['SFRH'].value 
    mtot = f_eagle['TotalMassFormed'].value  
    
    dt = age_bounds[:,2] - age_bounds[:,0]
    mtot = np.sum(dt * sfh[igal][:,i_metal]) * 1e9

    fig = plt.figure(figsize=(4,4))
    sub = fig.add_subplot(111)
    sub.plot(age_bounds[:,1], np.cumsum(dt * sfh[igal][:,i_metal] * 1e9), c='k', ls=':', lw=1, label='SFH')
    sub.plot(age_bounds[:,1], (dt * sfh[igal][:,i_metal] * 1e9), c='C0', lw=1, label='SFH')
    sub.plot([0., 14.], [mtot, mtot], c='k', ls='--', label='Input')
    sub.plot([0., 14.], [mmed, mmed], c='C1', ls='--', label='Firefly')
    print 10**(ffly_prop['age_massW']), 10**(ffly_prop['age_lightW'])
    sub.vlines(ffly_prop['age_massW'], 1e7, 1e10, linestyle='--', color='C1') 
    sub.vlines(ffly_prop['age_lightW'], 1e7, 1e10, linestyle=':', color='C1') 
    sub.set_xlabel('$t_\mathrm{cosmic}$ [Gyr]', fontsize=25) 
    sub.set_xlim([0., 14.]) 
    sub.set_ylabel('$M_\mathrm{tot}$', fontsize=25)  
    sub.set_yscale('log') 
    sub.set_ylim([1e7, 1e10]) 
    fig.savefig(''.join([UT.fig_dir(), 'simple_firefly.', str(igal), '.metal', str(i_metal), '.z', str(zred), '.png']), 
        bbox_inches='tight') 
    return None 


if __name__=='__main__':
    for i in range(1,5):
        #simple_spectra(i, i_metal=0, zred=0.01)
        simple_firefly(i, i_metal=0, zred=0.01)
