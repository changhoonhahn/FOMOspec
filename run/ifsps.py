'''
'''
import numpy as np 
from fomospec import fitters

def ifsps(): 
    ifsps = fitters.iFSPS()
    w, spec = ifsps.model(np.array([1e9, 0.019, 10., 1., 2.]), zred=0.1)
    print w, spec


if __name__=="__main__": 
    ifsps()
