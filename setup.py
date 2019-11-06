#!/usr/bin/env python
from setuptools import setup

__version__ = '0.1'

setup(name = 'FOMOspec',
      version = __version__,
      python_requires='>3.5.2',
      description = 'Forward Modeling Observed Spectra',
      author='ChangHoon Hahn',
      author_email='changhoonhahn@lbl.gov',
      requires = ['numpy', 'matplotlib', 'scipy'],
      provides = ['fomospec'],
      packages = ['fomospec']
      )
