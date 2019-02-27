'''
'''
import os 
import numpy as np 
from fomospec import fitters
from fomospec import util as UT  
# -- 
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


def ifsps(): 
    ifsps = fitters.iFSPS(model_name='vanilla')
    # theta: mass, Z, t_age, dust2, tau 
    tt_fid = np.array([1e9, 0.019, 10., 0., 2.])
    tt_tau_p = np.array([1e9, 0.019, 10., 0., 3.])
    tt_tau_pp = np.array([1e9, 0.019, 10., 0., 5.])
    tt_tau_ppp = np.array([1e9, 0.019, 10., 0., 7.])
    ws, specs = ifsps.model(np.array([tt_fid, tt_tau_p, tt_tau_pp, tt_tau_ppp]), zred=0.1)

    tt_tage_p = np.array([1e9, 0.019, 11., 0., 2.])
    tt_tage_pp = np.array([1e9, 0.019, 12., 0., 2.])
    tt_tage_ppp = np.array([1e9, 0.019, 13., 0., 2.])
    _ws, _specs = ifsps.model(np.array([tt_fid, tt_tage_p, tt_tage_pp, tt_tage_ppp]), zred=0.1)

    tt_dust_p = np.array([1e9, 0.019, 10., 1., 2.])
    tt_dust_pp = np.array([1e9, 0.019, 10., 2., 2.])
    tt_dust_ppp = np.array([1e9, 0.019, 10., 3., 2.])
    __ws, __specs = ifsps.model(np.array([tt_fid, tt_dust_p, tt_dust_pp, tt_dust_ppp]), zred=0.1)

    fig = plt.figure(figsize=(10,12))
    sub = fig.add_subplot(311)
    for w, spec in zip(ws, specs): 
        sub.plot(w, spec)
    sub.text(0.95, 0.95, r'varying $\tau$', ha='right', va='top', transform=sub.transAxes, fontsize=20)
    sub.set_xlim(3e3, 1e4) 
    sub.set_ylim(0, 2e-17)
    sub = fig.add_subplot(312)
    for w, spec in zip(_ws, _specs): 
        sub.plot(w, spec, ls='--')
    sub.text(0.95, 0.95, r'varying $t_{\rm age}$', ha='right', va='top', transform=sub.transAxes, fontsize=20)
    sub.set_xlim(3e3, 1e4) 
    sub.set_ylim(0, 2e-17)
    sub = fig.add_subplot(313)
    for w, spec in zip(__ws, __specs): 
        sub.plot(w, spec, ls='--')
    sub.text(0.95, 0.95, 'varying dust2', ha='right', va='top', transform=sub.transAxes, fontsize=20)
    sub.set_xlabel('Wavelength [Angstrom]', fontsize=25)
    sub.set_xlim(3e3, 1e4) 
    sub.set_ylim(0, 2e-17)
    fig.savefig(os.path.join(UT.fig_dir(), 'ifsps.test.png'), bbox_inches='tight') 
    return None


def ifsps_prior(): 
    ifsps = fitters.iFSPS(model_name='vanilla')
    # theta: mass, Z, t_age, dust2, tau 
    tt_fid = np.array([1e9, 0.019, 10., 0., 2.])
    tt_tau_p = np.array([1e9, 0.019, 10., 0., 3.])
    tt_tau_pp = np.array([1e9, 0.019, 10., 0., 5.])
    tt_tau_ppp = np.array([1e9, 0.019, 10., 0., 1e3])
    
    for tt in [tt_fid, tt_tau_p, tt_tau_pp, tt_tau_ppp]: 
        print ifsps.lnPrior(tt) 
    return None


def ifsps_posterior(): 
    ifsps = fitters.iFSPS(model_name='vanilla')
    # theta: mass, Z, t_age, dust2, tau 
    tt_fid = np.array([1e9, 0.019, 10., 0., 2.])
    w, spec = ifsps.model(tt_fid, zred=0.1)
    spec_err = 0.1*spec

    tt_tau_p = np.array([1e9, 0.019, 10., 0., 3.])
    tt_tau_pp = np.array([1e9, 0.019, 10., 0., 5.])
    tt_tau_ppp = np.array([1e9, 0.019, 10., 0., 1e3])
    for tt in [tt_fid, tt_tau_p, tt_tau_pp, tt_tau_ppp]: 
        print ifsps.lnPost(tt, 0.1, spec, spec_err, outwave=w) 
    return None


if __name__=="__main__": 
    #ifsps()
    #ifsps_prior()
    ifsps_posterior()
