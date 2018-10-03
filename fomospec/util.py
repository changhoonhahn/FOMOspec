'''

General utility functions 

'''
import os
import numpy as np


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
    if os.path.isdir(code_dir()+'figs/'):
        return code_dir()+'figs/'
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

