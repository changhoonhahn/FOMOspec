'''

General utility functions 

'''
import os
import h5py
import numpy as np
import astropy.units as U 
from astropy.io import fits 
import astropy.constants as Const
from astropy.cosmology import FlatLambdaCDM


def readmyFirefly(f_ffly): 
    '''
    '''
    f = h5py.File(f_ffly, 'r') 
    output = {} 
    for k in f.keys(): 
        output[k] = f[k].value 
    props = {} 
    for k in f.attrs.keys(): 
        props[k] = f.attrs[k]
    return output, props


def readFirefly(f_ffly): 
    ''' Read in firefly fitting output file
    '''
    f = h5py.File(f_ffly, 'r')
    output = {} 
    for g in f.keys(): 
        if g != 'properties': 
            output[g] = f[g].value
            
    props = {} 
    for k in f['properties'].keys(): 
        props[k] = f['properties'][k].value
    return output, props


def readProspector(f_pros): 
    ef = h5py.File(f_pros, 'r')
    chain = ef['sampling']['chain'].value
    lnp = ef['sampling']['lnprobability'].value
    ef.close()
    return chain,lnp 


def readDESIspec(ffits): 
    ''' read DESI spectra fits file

    :params ffits: 
        name of fits file  
    
    :returns spec:
        dictionary of spectra
    '''
    fitobj = fits.open(ffits)
    
    spec = {} 
    for i_k, k in enumerate(['wave', 'flux', 'ivar']): 
        spec[k+'_b'] = fitobj[2+i_k].data
        spec[k+'_r'] = fitobj[7+i_k].data
        spec[k+'_z'] = fitobj[12+i_k].data
    return spec 


def check_env(): 
    if os.environ.get('FOMOSPEC_DIR') is None: 
        raise ValueError("set $FOMOSPEC_DIR environment varaible!") 
    if os.environ.get('FOMOSPEC_CODEDIR') is None: 
        raise ValueError("set $FOMOSPEC_DIR environment varaible!") 
    return None


def dat_dir(): 
    ''' directory that contains all the data files, defined by environment 
    variable $IQUENCH_DIR
    '''
    return os.environ.get('FOMOSPEC_DIR') 


def code_dir(): 
    if os.environ.get('FOMOSPEC_CODEDIR') is None: 
        raise ValueError("set $FOMOSPEC_CODEDIR environment varaible!") 
    return os.environ.get('FOMOSPEC_CODEDIR') 
    

def fig_dir(): 
    ''' directory to dump all the figure files 
    '''
    dir_fig = os.path.join(code_dir(), 'figs') 
    if os.path.isdir(dir_fig):
        return dir_fig 
    else: 
        raise ValueError("create figs/ folder in $FOMOSPEC_CODEDIR directory for figures")


def doc_dir(): 
    ''' directory for paper related stuff 
    '''
    if os.path.isdir(code_dir()+'doc/'):
        return code_dir()+'doc/'
    else: 
        raise ValueError("create doc/ folder in $FOMOSPEC_CODEDIR directory for documntation")


def Lsun(): 
    return 3.846e33  # erg/s


def parsec(): 
    return 3.085677581467192e18  # in cm


def to_cgs(): # at 10pc 
    lsun = Lsun()
    pc = parsec()
    return lsun/(4.0 * np.pi * (10 * pc)**2) 


def c_light(): # AA/s
    return 2.998e18


def jansky_cgs(): 
    return 1e-23

