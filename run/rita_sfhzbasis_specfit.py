import numpy as np
import matplotlib.pylab as plt
import astropy.io.fits as pyfits
import os
import corner
import emcee
from astropy.io import fits
from astropy.io import ascii
from astropy.cosmology import Planck13 as cosmo
import astropy.units as u
from scipy.interpolate import interp1d
import specfit_desi_module as spec_desi


plt.ion()

#paths
'''
    datadir = os.environ['HOME']+'/Dropbox/SpecMockFit_spectra/data/'
    workdirectory = os.environ['HOME']+'/Dropbox/SpecMockFit_spectra/'
    spectradir = workdirectory + 'spectra/'
    outdir = workdirectory+ 'emcee_results/'
    chainsdir = workdirectory + 'emcee_output/'
'''

basis_id_sfh = 'nowgt_lin_Nc4'
basis_id_zh= 'nowgt_lin_Nc2'
basis_id = 'nowgt_lin_2basis'

NMF_SFH_comp = np.loadtxt(datadir+'NMF_2basis_SFH_components_'+basis_id_sfh+'.txt') #the basis SFHs
NMF_Z_comp = np.loadtxt(datadir+'NMF_2basis_Z_components_'+basis_id_zh+'.txt') #the basis ZHs
Nbins = NMF_SFH_comp.shape[1]
Ncomp_sfh = NMF_SFH_comp.shape[0]
Ncomp_zh = NMF_Z_comp.shape[0]

#files with the wavelength and lookback time bins
sfh_t = np.loadtxt(datadir+'sfh_t_int.txt')
dt = np.loadtxt(datadir+'dt_int.txt')
wave = np.loadtxt(datadir+'wave.txt')

def get_T_ISM(wave, tau_ISM):
	#dust attenuation
	tau_lambda = tau_ISM * (wave/5500.0)**(-0.7)
	T_ISM = np.exp(-tau_lambda)
	return T_ISM

seed = 123456789

this_tau_ISM=0

ndim, nwalkers = Ncomp_sfh + Ncomp_zh + 1, 100
Nsamples = 1000
burnin = 500
ntemps = 5
SFF_maxprior = 50
dtau = 1e-4
tau_maxprior = this_tau_ISM + dtau
tau_minprior = np.max([this_tau_ISM - dtau, 0])
Z0_minprior = 0.00
zh_maxprior = 0.04 #prior on actual values of metallicity in zh
Z0_maxprior = zh_maxprior/np.max(NMF_Z_comp)
youngAge = 0.250 #definition of young stars, in Gyrs

print('priors on Z0: ', Z0_minprior, Z0_maxprior)

#cosmo = FlatLambdaCDM(H0=67.3 * u.km / u.s / u.Mpc, Om0=0.315)

##### theta holds the fitted parameters: [coefficients SFF , ZH 6:7, tauV]

#load SSP models
hdulist = fits.open(datadir + 'conroy_raw_miles.fits')
data = hdulist[1].data
age_m = np.array(data.field(0))[0] #in Gyr
flux_m = np.array(data.field(1))[0] * 3.839e26 #from L/Lsun to W/AA
wave_m = np.array(data.field(2))[0]
Z_m = np.array(data.field(3))[0]
models_norm = np.mean(flux_m[3,:,30])
flux_m = flux_m / models_norm


def to_simulation_grid(age_m, sfh_t, wave, flux_m, Z_m):
	# interpolate SSP models to lookbacktime grid of the NMF components
	flux = np.zeros([len(Z_m), len(wave), len(sfh_t)], dtype=(np.float32))
	for zz in range(0,len(Z_m)):
		for i in range(0,len(wave)):
			f = interp1d(age_m, flux_m[zz,i,:],bounds_error=False, fill_value=0)
			flux[zz,i,:] = f(sfh_t)
	return flux
flux = to_simulation_grid(age_m, sfh_t, wave, flux_m, Z_m)

def get_zh(cz):

	zh = np.dot(cz,NMF_Z_comp)
	return zh

def get_F(theta):

	c = theta[0:Ncomp_sfh]
	cz = theta[Ncomp_sfh:Ncomp_sfh+Ncomp_zh]

	#reconstruct SFF(t) and Z(t)
	sfh = np.dot(c, NMF_SFH_comp)
	zh = get_zh(cz)

	fg = np.zeros(len(wave))

	for i in range(0,len(sfh_t)):

		#find boundary za and zb for linear interpolation
		if zh[i] > 0:
			a = Z_m/zh[i]
			if np.any(a >= 1): #if > Z_m[0]
				zb = np.where(a >=1)[0][0]
				za = np.max([0,zb-1])
			else:
				zb = len(Z_m)-1
				za = zb
		else:
			za = 0
			zb = 0

		if zb > za: dz = (np.log10(Z_m[zb]) - np.log10(zh[i]))/(np.log10(Z_m[zb]) - np.log10(Z_m[za]))
		elif zb == za == 0: dz = 1
		elif zb == za == len(Z_m): dz = 0

		#print(np.shape(sfh), np.shape(flux),za, zb, a, zh)
		fg += sfh[i] * (flux[zb,:,i] * (1-dz) + flux[za,:,i]*dz)

	return fg

def get_F_dust(theta):
	F = get_F(theta)
	return F * get_T_ISM(wave, theta[Ncomp_sfh + Ncomp_zh])

def lnprior(theta):
	c = theta[0:Ncomp_sfh]
	cz = theta[Ncomp_sfh : Ncomp_sfh + Ncomp_zh]
	tau = theta[Ncomp_sfh + Ncomp_zh]

	#zh = get_zh(cz)

	if np.all( c < SFF_maxprior) and np.all(c > 0) and (tau_minprior < tau < tau_maxprior) and np.all(cz < Z0_maxprior) and np.all(cz > Z0_minprior) :
		return 0.0
	else:
		return -np.inf


def lnlike(theta, F, F_err):
	Fsample = get_F_dust(theta)
	if np.all(F_err > 0): return -0.5*np.sum( ((F-Fsample)**2.)/(F_err**2))
	else: return -0.5*np.sum( ((F-Fsample)**2.))

def lnprob(theta, F, F_err):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, F, F_err)

#def read_data(fnm):
#	hdu = fits.open(spectradir + fnm)
#	redshift = hdu[0].header['redshift']
#
#	data = hdu[1].data
#	wave_obs = np.array(data['wave'])
#	flux_obs_in = np.array(data['flux_nodust_noise'])
#	flux_obs_err_in = np.array(data['flux_nodust_noise_err']) #flux in W/AA/m^2
#
#	wave_obs = wave_obs[flux_obs_in > 0]
#	flux_obs_err_in = flux_obs_err_in[flux_obs_in > 0]
#	flux_obs_in = flux_obs_in[flux_obs_in > 0]
#
#	#shift to rest-frame
#	wave_gal = wave_obs / (1.0 + redshift)
#
#	dL = cosmo.luminosity_distance(redshift) #luminosity distance in meters
#	dL = dL.to(u.m)
#
#	#interpolate to models wavelength resolution - DO BETTER
#	flux_obs = np.interp(wave, wave_gal, flux_obs_in) *  (4*np.pi*dL.value**2*(1+redshift)) #convert to luminosity in W/AA
#	flux_obs_err = np.interp(wave, wave_gal, flux_obs_err_in) * (4*np.pi*dL.value**2*(1+redshift))
#
#	return wave, flux_obs, flux_obs_err

def read_data(fnm, desi=None, template=None, desi_dir=None):
	if desi_dir:
		wave_obs, flux_obs_in, flux_obs_err_in = spec_desi.parse_1D_spectra(desi_dir + fnm)
		redshift = spec_desi.get_redshift(desi_dir+fnm)
		flux_obs_in = flux_obs_in*1e-17*1e-7*1e4 #from 10-17 ergs/s/cm2/AA to W/AA/m^2
		flux_obs_err_in = flux_obs_err_in*1e-17*1e-7*1e4 #from 10-17 ergs/s/cm2/AA to W/AA/m^2
	if template:
		hdu = fits.open(spectradir + fnm)
		redshift = hdu[0].header['redshift']

		data = hdu[1].data
		wave_obs = np.array(data['wave'])
		flux_obs_in = np.array(data['flux_nodust_noise'])
		flux_obs_err_in = np.array(data['flux_nodust_noise_err']) #flux in W/AA/m^2

		wave_obs = wave_obs[flux_obs_in > 0]
		flux_obs_err_in = flux_obs_err_in[flux_obs_in > 0]
		flux_obs_in = flux_obs_in[flux_obs_in > 0]

	#shift to rest-frame
	wave_gal = wave_obs / (1.0 + redshift)

	dL = cosmo.luminosity_distance(redshift) #luminosity distance in meters
	dL = dL.to(u.m)

	#interpolate to models wavelength resolution - DO BETTER
	flux_obs = np.interp(wave, wave_gal, flux_obs_in) *  (4*np.pi*dL.value**2*(1+redshift)) #convert to luminosity in W/AA
	flux_obs_err = np.interp(wave, wave_gal, flux_obs_err_in) * (4*np.pi*dL.value**2*(1+redshift))

	return wave, flux_obs, flux_obs_err


def run_emcee(fnm, desi_dir=None):
	nwalkers0 = nwalkers #* ntemps
	print('Will use ' + str(nwalkers0) + ' walkers')

	#read in data to analyse,
	wave, y, y_err = read_data(fnm, desi_dir=desi_dir)

	#normalise to keep numbers small
	norm = np.mean(y)
	print('norm=', norm)
	y = y / norm
	y_err = y_err/norm

	#emcee analysis
	print('...running emcee for ',fnm,'...')
	idg = fnm.split('.fits')[0]
	print('idg = ' + idg)

	#starting point por each walker, sampling randomly from prior
	pos_SFF = [np.random.random(Ncomp_sfh)*SFF_maxprior for i in range(nwalkers0)]
	pos_Z0 = [np.random.random(Ncomp_zh)*Z0_maxprior for i in range(nwalkers0)]
	pos_tau = [np.random.random()*(tau_maxprior-tau_minprior) + tau_minprior for i in range(nwalkers0)]
	pos = np.vstack([np.array(pos_SFF).T, np.array(pos_Z0).T, pos_tau]).T

	sampler = emcee.EnsembleSampler(nwalkers0, ndim, lnprob, args=(y, y_err))
	sampler.run_mcmc(pos, Nsamples)

	samples = sampler.chain[:, burnin:, :].reshape((-1, ndim)) #remove burn-in, line up all chains from individual walkers

	#np.savetxt(chainsdir + 'emcee_samples_'+id+'.txt', samples)
	if Ncomp_sfh == 5:
		fig = corner.corner(samples, labels=['NMF0', 'NMF1', 'NMF2', 'NMF3', 'NMF4', 'NMF_Z0', 'NMF_Z1', 'tauV'], quantiles=[0.16, 0.5, 0.84])
	if Ncomp_sfh == 4:
		fig = corner.corner(samples, labels=['NMF0', 'NMF1', 'NMF2', 'NMF3', 'NMF_Z0', 'NMF_Z1', 'tauV'], quantiles=[0.16, 0.5, 0.84])
	outfile = workdirectory+'/plots/corner_plot_'+idg+'_'+basis_id+'_lin.pdf'
	plt.savefig(outfile, bbox_inches='tight')

	samples[:,0:Ncomp_sfh] = samples[:,0:Ncomp_sfh] * norm / models_norm #save samples of SFF components in stellar mass units
	np.savetxt(chainsdir + 'emcee_samples_'+'_'+idg+'_'+basis_id+'_lin.txt', samples)

	return samples
	#return(norm, samples)	
