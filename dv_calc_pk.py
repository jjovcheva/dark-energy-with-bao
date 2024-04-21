import os, sys, emcee, corner
import numpy as np
import pandas as pd
from colossus.cosmology import cosmology
import scipy.optimize as op
import scipy.interpolate as interp
from scipy.constants import speed_of_light
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from classy import Class
from statsmodels.stats.moment_helpers import cov2corr
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Serif"],
})

def calc_cov(ell, cap, N):
    '''
    Calculate the covariance matrix from simulated catalogues.
    
    Parameters
    -----------
    ell : degree of multipole
    cap : SGC or NGC
    N : number of mock power spectra
    '''

    list_of_pks = []
    for i in range(1, N):
        pk_file = 'output/pk_sims/%s/pk0_sim_%s_%d.txt' % (cap, cap, i)
        
        # Load saved power spectra as dataframe.
        df = np.loadtxt(pk_file, skiprows=13, usecols=[0,3,5])
        df = pd.DataFrame(data=df, columns=['k_cen', 'Re{pk%s_raw}' %ell, 'Re{pk%s_shot}' %ell])
            
        # Subtract shot noise.    
        pk = df['Re{pk%s_raw}' %ell] - df['Re{pk%s_shot}' %ell]
        
        list_of_pks.append(pk)
        
    return np.cov(np.vstack(list_of_pks).T)  

def get_smooth_model(parameters, k, templates):
    ''' 
    Combine a noBAO model with polynomials and linear bias
    to make the fit independent of the broadband. 
    
    Parameters
    ----------
    parameters : p0, p1, etc. 
    k : wavenumber
    templates : models (here, without BAO)
    '''
    polynomials = parameters[1]/k**3 + parameters[2]/k**2 + parameters[3]/k + parameters[4] + parameters[5]*k
    return parameters[0]*parameters[0]*templates['noBAO'](k) + polynomials

def get_shifted_model(parameters, k, templates):
    ''' 
    Calculate a model including a scaling parameter.
    
    Parameters
    ----------
    parameters: p0, p1, etc.
    k: wavenumber
    templates: models (here, without BAO)
    '''
    model = get_smooth_model(parameters, k, templates)
    return model*(1. + (templates['os_model'](k/parameters[6]) - 1.)*np.exp(-0.5*k**2*parameters[7]**2))
    
def within_priors(parameters):
    '''Test for priors.'''
    if len(parameters) > 6. and abs(parameters[6] - 1.) > 0.2:
        return False
    elif len(parameters) > 6. and (parameters[7] < 0. or parameters[7] > 20.):
        return False
    else:
        return True
        
def calc_cov_bootstrap(ell, cap, N, B=3000):
    '''
    Calculate the bootstrap-averaged covariance matrix from simulated catalogues.

    Parameters
    ----------
    ell : degree of multipole
    cap : SGC or NGC
    N : number of mock power spectra
    B : number of bootstrap resamples (default=1000)

    Returns
    -------
    The averaged covariance matrix over B bootstrap resamples.
    '''
    list_of_pks = []
    for i in range(1, N+1):
        pk_file = f'output/pk_sims/{cap}/pk0_sim_{cap}_{i}.txt'
        df = np.loadtxt(pk_file, skiprows=13, usecols=[0,3,5])
        pk = df[:,1] - df[:,2] 
        list_of_pks.append(pk)

    cov_matrices = []
    for b in range(B):
        indices = np.random.randint(0, N, N)  # Resample indices with replacement
        resampled_pks = [list_of_pks[idx] for idx in indices]
        cov = np.cov(np.array(resampled_pks).T)
        cov_matrices.append(cov)

    # Average the covariance matrices
    averaged_cov = np.mean(cov_matrices, axis=0)
    return averaged_cov

def read_pk(ell, cap, tag):
    '''
    Read in the power spectrum.
    
    Parameters
    ----------
    ell : multipole degree
    cap : 'NGC' or 'SGC'
    tag : CMASS, LOWZ, etc.
    '''

    pk_dir = './output/measurements'
    filename = os.path.join(pk_dir, 'pk%s_%s_%s.txt' % (ell, cap, tag))

    df = np.loadtxt(filename, skiprows=13, usecols=[0,3,5])
    df = pd.DataFrame(data=df, columns=['k_cen', 'Re{pk%s_raw}' %ell, 'Re{pk%s_shot}' %ell])
    a, b, c = [], [], []
    a.append(df['k_cen'].to_list())
    b.append(df['Re{pk%s_raw}' %ell].to_list())
    c.append(df['Re{pk%s_shot}' %ell].to_list())
    
    return df

def pk_wiggle(krange):
    cosmo = cosmology.setCosmology('planck18')
    
    pk = cosmo.matterPowerSpectrum(krange, z=0.0, model='camb')
    return pk

def pk_no_bao(krange):
    cosmo = cosmology.setCosmology('planck18')

    pk = cosmo.matterPowerSpectrum(krange, z = 0.0, model='eisenstein98_zb')
    return pk
        
def calc_chi2(parameters, data, templates, func):
    '''Calculate the chi-squared value.'''
    if within_priors(parameters):
        chi2 = 0.
        # Loop over all datasets which are fit together
        for dataset in data:
            model = func(parameters, dataset['k'], templates)
            diff = (model - dataset['pk'])
            chi2 += np.dot(diff, np.dot(dataset['cov_inv'], diff))
        return chi2
    else:
        return 100000.

def get_oscillation(krange):
    ''' Get an oscillation only power spectrum '''
    cov_inv = np.identity(len(krange))
    start = [1., 0., 0., 0., 0., 0.]
    result = op.minimize(calc_chi2, start, args=( [{ 'k': krange, 'pk': pk_wiggle(krange), 'cov_inv': cov_inv }],\
     { 'noBAO': pk_no_bao }, get_smooth_model))
    yolin = pk_wiggle(krange)/get_smooth_model(result["x"], krange, { 'noBAO': pk_no_bao })
    return interp.interp1d(krange, yolin)

def get_percentiles(chain, labels):
    '''Calculate constraints and uncertainties from MCMC chain.'''
    per = np.percentile(chain, [50., 15.86555, 84.13445, 2.2775, 97.7225], axis=0)
    per = np.array([per[0], per[0]-per[1], per[2]-per[0], per[0]-per[3], per[4]-per[0]])
    
    with open('output/percentiles', 'w') as f:
        for i in range(len(per[0])):
            output_line = "%s = %f +%f -%f +%f -%f\n" % (labels[i], per[0][i], per[1][i], per[2][i], per[3][i], per[4][i])
            f.write(output_line)
    return per

def get_loglike(parameters, data, templates, func):
    '''Get log likelihood.'''
    return -0.5*calc_chi2(parameters, data, templates, func)

def gelman_rubin_convergence(within_chain_var, mean_chain, chain_length):
    ''' Calculates the Gelman-Rubin diagnostic.'''

    Nchains = within_chain_var.shape[0]
    dim = within_chain_var.shape[1]

    mean_all = np.mean(mean_chain, axis=0)  # Overall mean
    W = np.mean(within_chain_var, axis=0)  # Within-chain variance

    # Between-chain variance (vectorized for potential efficiency)
    B = chain_length * np.sum((mean_chain - mean_all) ** 2, axis = 0) / (Nchains - 1) 

    est_var = (1 - 1/chain_length) * W + B/chain_length  # Estimated overall variance
    Rhat = np.sqrt(est_var / W)  # Potential scale reduction factor 

    print('Parameter means: ', mean_all)
    return Rhat

def prep_gelman_rubin(sampler):
    '''
    Prepare chains for testing Gelman-Rubin convergence.
    
    Parameters
    ----------
    sampler : sampler containing information about the MCMC chains.
    
    Return
    ------
    within_chain_var : variance for each parameter across all chains.
    mean_chain : mean parameter values across all chains.
    '''
    # Dimensionality of the parameter space.
    dim = sampler.chain.shape[2]
    
    # Length of each chain.
    chain_length = sampler.chain.shape[1]
    
    # Extract samples from chains.
    # Discard a portion of samples from burn-in period.
    chainsamples = sampler.chain[:, int(chain_length/2):, :].reshape((-1, dim))
    
    # Calculate variance for each parameter across all chains.
    within_chain_var = np.var(chainsamples, axis=0)
    
    # Calculate mean parameter values across all chains.
    mean_chain = np.mean(chainsamples, axis=0)
    
    return within_chain_var, mean_chain
    
def runMCMC(start_params, data, templates):
    ndim = len(start_params)
    num_chains = 4
    num_walkers = 20
    check_interval = 400
    minlength = 2000
    epsilon = 0.01
    labels = ['b', 'A1', 'A2', 'A3', 'A4', 'A5', 'alpha', 'sigmaNL']
    expected_error = [0.1, 1., 1., 1., 1., 1., 0.05, 0.1]

    # Initialise walker positions.
    pos = [start_params + (2.*np.random.random_sample((ndim,)) - 1.) * expected_error for _ in range(num_walkers*num_chains)]
    pos = np.array(pos).reshape((num_chains, num_walkers, ndim))
    
    samplers = [
        emcee.EnsembleSampler(num_walkers, ndim, get_loglike, args=(data, templates, get_shifted_model))
        for _ in range(num_chains)
    ] # Set up samplers.
    
    within_chain_var = np.zeros((num_chains, ndim))
    mean_chain = np.zeros((num_chains, ndim))
    rhat = np.arange(ndim, dtype=float)
    rhat.fill(2.)
    total_steps = 0
    max_iat = 0  # Initialize maximum Integrated Autocorrelation Time (IAT).
    
    chainstep = minlength
    print("Running MCMC...")
    while any(abs(1. - rhat) > epsilon):
        total_steps += chainstep
        for j, sampler in enumerate(samplers):
            pos[j], _, _ = sampler.run_mcmc(pos[j], check_interval, progress=True, store=True)
            within_chain_var[j], mean_chain[j] = prep_gelman_rubin(sampler)

        # Calculate scale reduction parameter.
        rhat = gelman_rubin_convergence(within_chain_var, mean_chain, total_steps // 2)
        
        chainstep = check_interval
        print('Steps: ', total_steps)
        print('Scale reduction parameter: ', rhat)
        
        # Calculate the maximum autocorrelation time across all parameters and chains.
        try:
            iat = max(np.max(sampler.get_autocorr_time()) for sampler in samplers)
            max_iat = max(max_iat, iat)
        except emcee.autocorr.AutocorrError:
            iat = np.inf

        # Check for convergence (with extra criterion of minimum chain length).
        if total_steps > minlength and total_steps > 50 * max_iat and np.all(rhat < 1 + epsilon):
            print(f"Convergence achieved at iteration: {total_steps}")
            break

    # Flatten the chains, discarding burn-in.
    burn_in = int(0.5 * total_steps)
    mergedsamples = np.concatenate([s.get_chain(discard=burn_in, flat=True) for s in samplers])
        
    if labels:
        np.savetxt("mcmc_samples.txt", mergedsamples, header=" ".join(labels))
        
    return get_percentiles(mergedsamples, labels)

def calculate_DV(zmean, cosmo):
    '''
    Calculate the spherically-averaged distance.

    Parameters
    ----------
    z : float
        Redshift.
    cosmo : astropy.cosmology.FLRW
        Astropy cosmology object.

    Returns
    -------
    D_V : float
        The spherically-averaged distance measure D_V in Mpc.
    '''
    # Calculate H(z) in units of (km/s)/Mpc
    H_z = cosmo.efunc(zmean)*cosmo.H0*cosmo.h

    # Convert speed of light from m/s to km/s
    c_km_s = (speed_of_light/1000.)
    
    DC = (1.+zmean)*cosmo.angular_diameter_distance(zmean)/cosmo.h
    DVfid = ( DC**2*(zmean*c_km_s/H_z) )**(1./3.)

    return DVfid.value

def estimate_dv(tag, ell=0):
    '''Estimate the BAO scale.'''
    ngc_data = read_pk(ell, 'NGC', tag)
    sgc_data = read_pk(ell, 'SGC', tag)

    cov_ngc = calc_cov_bootstrap(ell, 'NGC', 500)
    cov_sgc = calc_cov_bootstrap(ell, 'SGC', 500)
    
    inv_cov_ngc = inv(calc_cov_bootstrap(ell, 'NGC', 500)) 
    inv_cov_sgc = inv(calc_cov_bootstrap(ell, 'SGC', 500))  

    pk_ngc = ngc_data['Re{pk0_raw}'] - ngc_data['Re{pk0_shot}']
    pk_sgc = sgc_data['Re{pk0_raw}'] - sgc_data['Re{pk0_shot}']
    
    boss_data = [{'k': ngc_data['k_cen'],\
                  'pk': pk_ngc,\
                  'label': 'BOSS NGC',\
                  'cov': cov_ngc,\
                  'cov_inv': inv_cov_ngc},
                 
                 {'k': sgc_data['k_cen'],\
                  'pk': pk_sgc,\
                  'label': 'BOSS SGC',\
                  'cov': cov_sgc,\
                  'cov_inv': inv_cov_sgc}]
    
    krange = np.arange(0.001, 0.5, 0.001)
    os_model = get_oscillation(krange)
    start = [2.37, -0.076, 38., -3547., 15760., -22622., 1., 9.41]
    result = op.minimize(calc_chi2, start, args=(boss_data, { 'noBAO': pk_no_bao, 'os_model': os_model },\
     get_shifted_model))
    print("Optimised parameters: ", result.x)
        
    krange = ngc_data['k_cen']
    best_fit_model = get_shifted_model(result["x"], krange, { 'noBAO': pk_no_bao, 'os_model': os_model })/get_smooth_model(result["x"], krange, { 'noBAO': pk_no_bao })
    boss_data[0]['ratio'] = boss_data[0]['pk']/get_smooth_model(result["x"], boss_data[0]['k'], { 'noBAO': pk_no_bao })
    boss_data[0]['ratio_err'] = np.sqrt(np.diagonal(boss_data[0]['cov']))/get_smooth_model(result["x"], boss_data[0]['k'], { 'noBAO': pk_no_bao })
    boss_data[1]['ratio'] = boss_data[1]['pk']/get_smooth_model(result["x"], boss_data[1]['k'], { 'noBAO': pk_no_bao })
    boss_data[1]['ratio_err'] = np.sqrt(np.diagonal(boss_data[1]['cov']))/get_smooth_model(result["x"], boss_data[1]['k'], { 'noBAO': pk_no_bao })

    per = runMCMC(result.x, boss_data, { 'noBAO': pk_no_bao, 'os_model': os_model })
    zmean = np.mean([0.60667086, 0.6059798])
    if per is not None:
        DV = calculate_DV(zmean, cosmo) * per[:,6] 
        print("DV = %f +%f -%f +%f -%f\n" % (DV[0], DV[1], DV[2], DV[3], DV[4]))
    else:
        print("Problem with MCMC chain")
    return 

if __name__ == "__main__":
    estimate_dv('CMASSLOWZTOT') 
