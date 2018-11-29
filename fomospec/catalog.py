'''



'''
import h5py 
import numpy as np 
from astropy.io import fits
from astropy.table import Table
# -- fomospec -- 
from . import util as UT 


class LGal(object): 
    ''' interface with LGal catalogs 
    '''
    def __init__(self): 
        self._dir_lgal = ''.join([UT.dat_dir(), 'Lgal/']) 

    def GalInput(self, galid): 
        ''' read in input star-formation and chemical enrichment histories
        '''
        f_input = ''.join([self._dir_lgal, 'gal_inputs/', 
            'gal_input_' + str(galid) + '_BGS_template_FSPS_uvmiles.csv']) 
        gal_input = Table.read(f_input, delimiter=' ')
        return gal_input

    def Spectra(self, galid, type='source', lib='bc03'): 
        ''' Read in spectra given galid
        '''
        # source spectra file with meta data 
        f_source = fits.open(self._Fspec(galid, 'source', lib)) 
        
        # get meta data of spectra 
        hdr = f_source[0].header
        meta = {}
        for k in hdr.keys(): 
            meta[k] = hdr[k]

        spec = {} 
        if type == 'source':  
            # source spectra 
            specin = f_source[1].data
            spec['wave'] = specin['wave']
            spec['flux'] = specin['flux_nodust_nonoise'] * 1e20
            spec['flux_unc'] = None 

        elif type == 'desibgs': 
            # desi-like spectra (in desiIO format) 
            import desispec.io as desiIO
            f_desi = self._Fspec(galid, type, lib)#''.join([self._dir_lgal, 'spectra/', 'desi_out_', f_spec])
            spec_desi = desiIO.read_spectra(f_desi)

            spec['wave'] = np.concatenate([spec_desi.wave[b] for b in ['b', 'r', 'z']])
            spec['flux'] = np.concatenate([spec_desi.flux[b][0] for b in ['b', 'r', 'z']]) # 10-17 ergs/s/cm2/AA
            spec['flux_unc'] = np.concatenate([spec_desi.ivar[b][0]**-0.5 for b in ['b', 'r', 'z']])
        
        return spec, meta
    
    def SpecFit(self, galid, type='source', lib='bc03', fit='firefly', **fitkwargs): 
        ''' Read in output files from different spectral fitters
        '''
    
        if fit == 'firefly': 
            if 'dust' not in fitkwargs.keys(): 
                raise ValueError("specify `dust` kwarg") 
        else: 
            raise NotImplementedError 

        if fit == 'firefly': 
            f_spec = self._Fspec(galid, type, lib)
            return self._readFirefly(f_spec, dust=fitkwargs['dust']) 

    def _readFirefly(self, f_spec, dust='hpf_only'): 
        ''' read in FireFly output file 
        '''

        f_spec = '.'.join(f_spec.split('/')[-1].split('.')[:-1])+'.hdf5'

        f_ffly = ''.join([self._dir_lgal, 'spectra/',
            'firefly.m11.MILES.imf_cha.dust_', dust, '.', f_spec]) 
        f = h5py.File(f_ffly, 'r')
        
        output = {} 
        for g in f.keys(): 
            if g != 'properties': 
                output[g] = f[g].value
                
        props = {} 
        for k in f['properties'].keys(): 
            props[k] = f['properties'][k].value
        return output, props

    def _Fspec(self, galid, type, lib): 
        ''' spectra file
        '''
        if lib == 'bc03': 
            str_lib = 'BC03_Stelib'
        else: 
            raise NotImplementedError
        f_spec = ''.join(['gal_spectrum_', str(galid), '_BGS_template_', str_lib, '.fits']) 
        f_source = ''.join([self._dir_lgal, 'templates/', f_spec])

        if type == 'source': 
            return f_source 
        elif type == 'desibgs': 
            return ''.join([self._dir_lgal, 'spectra/', 'desi_out_', f_spec])


'''
other catalogs here
'''
