'''

plots for the GQP spectral challenge


'''
import h5py 
import numpy as np 
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


def testGalIDs(): 
    ''' get gal IDs for test set of LGal SAM objects
    '''
    # read in spectral challenge test set filenames set by Rita 
    f_test = ''.join([UT.dat_dir(), 'spectral_challenge/', 'lgal_filenames_testset_BC03_Stellib.txt']) 
    fnames_test = np.loadtxt(f_test, unpack=True, dtype='S', skiprows=1)  
    # get gal ID's from the filename 
    galid_test = [int(fname.split('_')[2]) for fname in fnames_test]
    return galid_test


if __name__=="__main__":
    mFF_LGal_nodust(imf='chabrier', iobs=0, obs_sampling='spacefill')
