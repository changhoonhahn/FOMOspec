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


def mFF_LGal_nodust_Mstar(imf='chabrier', iobs=0, obs_sampling='spacefill'): 
    ''' Comparing Mformed inferred from myFirefly to 
    input Lgal Mformed 
    '''
    f_source = lambda gid: ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'LGAL.', str(gid), '.FSPS.nodust.imf_', imf, '.hdf5']) 
    # source spectrum firefly file 
    mff_source = lambda gid: ''.join([UT.dat_dir(), 'spectral_challenge/bgs/', 
        'myFF.LGAL.', str(gid), '.FSPS.nodust.imf_', imf, '.source.hdf5'])
    # bgs-like spectrum firefly file 
    mff_bgs = lambda gid: ''.join([UT.dat_dir(), 'spectral_challenge/bgs/mFF.LGAL.', str(gid), '.FSPS.nodust.imf_', imf, 
        '.BGS.', obs_sampling, '_obs', str(iobs), '.hdf5'])

    # gather M_inferred and M_input 
    galids = np.unique(testGalIDs())
    m_source, minf_source, minf_bgs = np.zeros(len(galids)), np.zeros(len(galids)), np.zeros(len(galids))
    merr_source, merr_bgs = np.zeros((2, len(galids))), np.zeros((2, len(galids))) 

    for i, galid in enumerate(galids): 
        # read in source spectrum FF file 
        ffly_out, ffly_prop = UT.readmyFirefly(mff_source(galid)) 
        minf_source[i] = 10**ffly_prop['logM_total']
        merr_source[0,i] = 10**ffly_prop['logM_total_up_1sig']
        merr_source[1,i] = 10**ffly_prop['logM_total_low_1sig']
        
        # read in BGS spectrum FF file 
        ffly_out, ffly_prop = UT.readmyFirefly(mff_bgs(galid)) 
        minf_bgs[i] = 10**ffly_prop['logM_total']
        merr_bgs[0,i] = 10**ffly_prop['logM_total_up_1sig']
        merr_bgs[1,i] = 10**ffly_prop['logM_total_low_1sig']
        
        # read in input M_total 
        lgal = h5py.File(f_source(galid), 'r') 
        m_source[i] = 10**lgal.attrs['logM_total']

    merr_source[0,:] -= minf_source
    merr_source[1,:] = minf_source - merr_source[1,:]
    
    merr_bgs[0,:] -= minf_bgs
    merr_bgs[1,:] = minf_bgs - merr_bgs[1,:]
    
    # scatter plot of log Mtotal (inferred) vs log Mtotal (input)
    fig = plt.figure(figsize=(7,7))
    sub = fig.add_subplot(111)
    sub.errorbar(m_source, minf_source, yerr=merr_source, fmt='.k', markersize=10, label='source (no noise)')
    sub.errorbar(m_source, minf_bgs, yerr=merr_bgs, fmt='.C1', markersize=5, label=r'bgs; $i_\mathrm{obs}='+str(iobs+1)+'$')
    sub.plot([1e8, 1e12], [1e8,1e12], c='k', ls='--')
    sub.set_xlabel(r'$M_\mathrm{total}^\mathrm{(input)}$ [$M_\odot$]', fontsize=25) 
    sub.set_xscale('log')
    sub.set_xlim([1e8, 1e12])
    sub.set_ylabel(r'$M_\mathrm{total}^\mathrm{(firefly)}$ [$M_\odot$]', fontsize=25) 
    sub.set_yscale('log')
    sub.set_ylim([1e8, 1e12])
    sub.legend(loc='upper left', markerscale=5, handletextpad=0, fontsize=20) 
    fig.savefig(''.join([UT.fig_dir(), 'mFF_LGal_nodust_Mstar.obs', str(iobs), '.png']), bbox_inches='tight') 

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
    mFF_LGal_nodust_Mstar(imf='chabrier', iobs=0, obs_sampling='spacefill')
