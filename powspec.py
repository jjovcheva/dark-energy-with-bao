import os, sys, ruamel.yaml, corner, emcee
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.io import fits
import dask.array as da
from astropy.cosmology import Planck18 as cosmo
import pandas as pd
import dask.dataframe as dd
from statsmodels.stats.moment_helpers import cov2corr

import scipy.optimize as op
import scipy.interpolate as interp
from scipy.integrate import quad
from scipy.constants import speed_of_light
from numpy.linalg import inv

import seaborn as sns

from triumvirate.catalogue import ParticleCatalogue
from triumvirate.logger import setup_logger
from triumvirate.parameters import ParameterSet
from triumvirate.parameters import fetch_paramset_template
from triumvirate.twopt import compute_powspec, compute_powspec_in_gpp_box

from classy import Class
import cosmotools

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Serif"],
})

yaml = ruamel.yaml.YAML()

cat_dir = 'input/catalogues'
cov_dir = 'output/covariances'

conv_file = 'cat_conv.yml'
param_file = 'input/params/pk_params.yml'

def fits_intel():
    '''
    Inspect columns from FITS catalogue.
    '''
    with open(conv_file) as fp:
        data = yaml.load(fp)
    
    cat_data = data['files']['data_catalogue']
    hdul = fits.open(os.path.join(data['directories']['catalogues'], cat_data))
    columns = hdul[1].columns
    
    return columns 

def read_data():
    '''
    Read in FITS data catalogues, cut for appropriate 
    redshift interval, and determine n(z) and systematic
    and FKP weights.
    
    Return
    ------
    cat_data: data catalogue cut for desired redshifts
    nz: mean background density
    ws: sample weights
    wc: clustering weights
    '''
    # Read filename from dictionary.
    with open(conv_file) as fp:
        pars = yaml.load(fp)
    
    cat_data = fits.open(os.path.join(pars['directories']['catalogues'], 
                pars['files']['data_catalogue']))
    
    cat_data = cat_data[1].data
        
    # Cut the catalogue for the appropriate redshifts.
    cat_data = cat_data[(cat_data['Z'] > 0.04) & (cat_data['Z'] < 1.1)]
        
    # Create combined systematic weight column.
    weight = cat_data['WEIGHT_SYSTOT'] * \
            (cat_data['WEIGHT_NOZ'] + cat_data['WEIGHT_CP'] - 1.0)
    ws = np.array(weight, dtype=np.float64)
    wc = np.array(cat_data['WEIGHT_FKP'], dtype=np.float64)
    nz = np.array(cat_data['NZ'], dtype=np.float64)

    return cat_data, nz, ws, wc

def read_ran():
    '''
    Read in FITS random catalogues, cut for appropriate 
    redshift interval, and determine n(z) and systematic
    and FKP weights.
    
    Return
    ------
    cat_data: data catalogue cut for desired redshifts
    nz: mean background density
    ws: sample weights
    wc: clustering weights
    '''
    # Read in the random catalogue.
    with open(conv_file) as fp:
        pars = yaml.load(fp)
        
    cat_ran = fits.open(os.path.join('input/catalogues', 
                pars['files']['rand_catalogue']))
    cat_ran = cat_ran[1].data
    
    # Cut the catalogue for the appropriate redshifts.
    cat_ran = cat_ran[(cat_ran['Z'] > 0.01) & (cat_ran['Z'] < 0.9)]
    
    # Create systematic weight column with all values equal to 1.
    ws = np.ones(shape=(cat_ran['WEIGHT_FKP'].size))
    wc = np.array(cat_ran['WEIGHT_FKP'], dtype=np.float32)
    nz = np.array(cat_ran['NZ'], dtype=np.float32)
    
    return cat_ran, nz, ws, wc

def read_dat_sim():
    '''
    Read in a simulated data catalogue as a dask
    dataframe, cut for the appropriate redshifts, 
    and add weight columns.
    '''
    with open(conv_file) as fp:
        pars = yaml.load(fp)
        
    # Create dask dataframe.
    dat_sim = dd.read_csv(
        os.path.join(pars['directories']['sims'], pars['files']['data_sim']),\
        header=0, 
        delim_whitespace=True,
        names=['RA', 'DEC', 'Z', 'dummy1', 'NZ', 'dummy2', 'veto', 'Weight']
        )
    
    # Cut catalogue for the appropriate redshifts.
    dat_sim = dat_sim.loc[(dat_sim['Z'] > 0.01) & (dat_sim['Z'] < 0.9)]
    dat_sim = dat_sim.loc[(dat_sim['veto'] > 0)]
    
    # Add weight and n(z) columns.
    dat_sim['wc'] = dat_sim['NZ'].map_partitions(lambda x: 1./(1. + 10000.*x))
    dat_sim['nz'] = dat_sim['NZ']
    dat_sim['ws'] = da.ones_like(dat_sim['NZ'])
    
    return dat_sim
    
def read_ran_sim(): 
    '''
    Read in a simulated random catalogue as a dask
    dataframe, cut for the appropriate redshifts, and 
    add weight columns.
    '''
    # Read in the simulated random catalogues.
    with open(conv_file) as fp:
        pars = yaml.load(fp)
        
    # Create dask dataframe.
    ran_sim = dd.read_csv(
        os.path.join(pars['directories']['sims'], pars['files']['rand_sim']),\
        header=0, 
        delim_whitespace=True,
        names=['RA', 'DEC', 'Z', 'NZ', 'dummy1', 'veto', 'Weight']
        )

    # Cut catalogue for the appropriate redshifts.
    ran_sim = ran_sim.loc[(ran_sim['Z'] > 0.01) & (ran_sim['Z'] < 0.9)]
    ran_sim = ran_sim.loc[(ran_sim['veto'] > 0)]
    
    # Add weight and n(z) columns.
    ran_sim['wc'] = ran_sim['NZ'].map_partitions(lambda x: 1./(1. + 10000.*x))
    ran_sim['nz'] = ran_sim['NZ']
    ran_sim['ws'] = da.ones_like(ran_sim['NZ'])
    
    return ran_sim

def calc_comoving_distance(redshift):
    '''Calculate the comoving distance in Mpc given a redshift.'''
    return cosmo.comoving_distance(redshift).to('Mpc').value

def conv_csv(cat_type, cap):
    '''
    Convert simulated CSV-format catalogues to a form 
    that is compatible with Triumvirate. Save to a new 
    file in a separate directory.
    
    Parameters
    ----------
    cat_type: 'dat' or 'ran'
    cap: NGC or SGC
    '''
    with open(conv_file) as fp:
        pars = yaml.load(fp)

    if cat_type == 'dat':
        # Read in original CSV data catalogue.
        cat_name = pars['files']['data_sim']
        cat_data = read_dat_sim()
        
    if cat_type == 'ran':
        # Read in the original CSV random catalogue.
        cat_name = pars['files']['rand_sim']
        cat_data = read_ran_sim()

    # Pick out RA and Dec from original file.
    ra, dec = cat_data['RA'], cat_data['DEC']

    # Convert RA, Dec to radians.
    ra = da.deg2rad(cat_data['RA'].compute())
    dec = da.deg2rad(cat_data['DEC'].compute())    

    # Convert RA, Dec to Cartesian coordinates on the unit sphere.
    x = da.cos(dec) * da.cos(ra)
    y = da.cos(dec) * da.sin(ra)
    z = da.sin(dec)
    
    x = da.from_array(x, chunks=x.shape)
    y = da.from_array(y, chunks=y.shape)
    z = da.from_array(z, chunks=z.shape)

    # Multiply Cartesian coordinates on unit sphere by comoving distance
    # to obtain true Cartesian coordinates.
    r = cat_data['Z'].map(calc_comoving_distance, meta=('redshift', 'f8'))
    r = r.to_dask_array(lengths=True)

    cat_data['x'], cat_data['y'], cat_data['z'] = r * x, r * y, r * z
    
    cat_data = cat_data[['x', 'y', 'z', 'nz', 'ws', 'wc']]
    with open('new_mocks/%s/%s.csv' % (cap, cat_name), 'w') as f:
        np.savetxt(f, cat_data, header='x y z nz ws wc')
        f.close()
    
    return 

def conv_fits(cat_type):
    '''
    Convert FITS-format catalogues to a form that is 
    compatible with Triumvirate. Save to a new file in 
    a separate directory.
    
    Parameters
    ----------
    cat_type: 'dat' or 'ran'
    
    Return
    ------
    cat_data: data in Triumvirate-compatible form
    '''    
    with open(conv_file) as fp:
        conv = yaml.load(fp)
        
    if cat_type == 'dat':
        # Read in original FITS data catalogue.
        cat_name = conv['files']['data_catalogue']
        cat_data, nz, ws, wc = read_data()
                        
    elif cat_type == 'ran':
        # Read in original FITS random catalogue.
        cat_name = conv['files']['rand_catalogue']
        cat_data, nz, ws, wc = read_ran()
        
    # Pick out RA, Dec, and redshift from original FITS file.
    ra = cat_data['RA']
    dec = cat_data['DEC']
    redshift = cat_data['Z']
    
    # Convert RA, Dec to radians.
    ra, dec, redshift = da.broadcast_arrays(ra, dec, redshift)
    ra = da.deg2rad(ra)
    dec = da.deg2rad(dec)    
    
    # Convert RA, Dec to Cartesian coordinates on the unit sphere.
    x = da.cos(dec) * da.cos(ra)
    y = da.cos(dec) * da.sin(ra)
    z = da.sin(dec)

    # Multiply Cartesian coordinates on unit sphere by comoving distance
    # to obtain true Cartesian coordinates.
    r = cosmo.comoving_distance(redshift)
    x, y, z = r * x, r * y, r * z
    x, y, z = np.array(x, dtype=np.float32), \
              np.array(y, dtype=np.float32), \
              np.array(z, dtype=np.float32) 
    
    # Save data to new file.
    cat_data = [x, y, z, nz, ws, wc]
    with open('input/new_cats/%s' % cat_name, 'w') as f:
        np.savetxt(f, np.column_stack(cat_data), header='x y z nz ws wc')
        f.close()
                   
    return cat_data

def calc_pk(pars):
    '''Calculate the power spectrum.'''
    
    # Read in the data catalogue.
    cat_data = ParticleCatalogue.read_from_file(
        "{}/{}".format(
            pars['directories']['catalogues'],
            pars['files']['data_catalogue']
        ),
        names=['x', 'y', 'z', 'nz', 'ws', 'wc']
    )
    
    # Read in the random catalogue.
    cat_rand = ParticleCatalogue.read_from_file(
        "{}/{}".format(
            pars['directories']['catalogues'],
            pars['files']['rand_catalogue']
        ),
        names=['x', 'y', 'z', 'nz', 'ws', 'wc']    
    )
    
    # Compute the power spectrum and save to text file.
    measurements = compute_powspec(
        cat_data, cat_rand, paramset=pars, save='.txt'
        )

def process_data(ell, cap, tag):
    '''
    Calculate power spectra for galaxy data.
    
    Parameters
    ----------
    ell: multipole degree
    cap: NGC or SGC
    tag: CMASS, LOWZ, etc.
    '''
    # Read in the original random catalogue.
    with open(conv_file) as fp1:
        conv = yaml.load(fp1)    
       
    if not os.path.isfile(os.path.join(conv['directories']['new_cats'], conv['files']['data_catalogue'])):
        conv_fits('dat')
    else:
        pass
    
    if not os.path.isfile(
        os.path.join(conv['directories']['new_cats'], conv['files']['rand_catalogue'])):
        conv_fits('ran')
    else:
        pass
    
    with open('input/params/pk_params.yml') as fp2:
        pars = yaml.load(fp2) 
    
    # Update parameter file.
    pars['files']['data_catalogue'] = conv['files']['data_catalogue']
    pars['files']['rand_catalogue'] = conv['files']['rand_catalogue']
    pars['degrees']['ELL'] = ell
    pars['tags']['output'] = '_%s_%s' % (cap, tag)
    
    with open('input/params/pk_params.yml', 'w+') as fp2:
        yaml.dump(pars, fp2)
    
    # Calculate power spectrum.
    pars = ParameterSet(param_filepath='input/params/pk_params.yml')
    calc_pk(pars)
    
def calc_cov(ell, cap, N):
    '''
    Calculate the covariance matrix from simulated catalogues.
    
    Parameters
    -----------
    ell: degree of multipole
    cap: SGC or NGC
    N: number of mock power spectra
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

def process_sims(ell, cap, N):
    '''
    Convert the simulated catalogues, calculate power
    spectra, and calculate covariance matrix.
    
    Parameters
    ----------
    ell: degree of multipole
    cap: SGC or NGC
    N: number of mock power spectra
    '''
    
    # Test whether the power spectra are already available.
    # If so, calculate the covariance matrix directly.
    try:
        return calc_cov(cap, N)
    except:
        pass
    
    # Read in the original random catalogue.
    with open(conv_file) as fp1:
        conv = yaml.load(fp1)
        
    # Loop through all of the data sims.
    for i in range(1, N):
        pk_file = 'output/pk_sims/%s/pk%s_sim_%s_%d.txt' % (cap, ell, cap, i)
        # print('Calculating %s... ' % pk_file)

        # Only calculate power spectra which are not already on disk.
        if not os.path.isfile(pk_file):
            # Check whether the data catalogues have been converted.
            converted_ran_file = \
                'new_mocks/%s/Patchy-Mocks-Randoms-DR12%s-COMPSAM_V6C_x10.dat.csv' %(cap, cap)
    
            # Check if the random catalogue has already been converted.
            # If not, convert the catalogue to a usable form.
            if not os.path.isfile(converted_ran_file):
                conv['files']['rand_sim'] = \
                    'Patchy-Mocks-Randoms-DR12%s-COMPSAM_V6C_x10.dat' %cap
                with open(conv_file, 'w+') as fp1:
                    yaml.dump(conv, fp1)
                    
                conv_csv('ran', cap)  
                
            if not os.path.isfile(
                os.path.join(conv['directories']['new_sims'], 
                'Patchy-Mocks-DR12%s-COMPSAM_V6C_%0.4d.dat.csv' % (cap, i))):
                
                conv['files']['data_sim'] = 'Patchy-Mocks-DR12%s-COMPSAM_V6C_%0.4d.dat' % (cap, i)
               
                with open(conv_file, 'w+') as fp1:
                    yaml.dump(conv, fp1)
                conv_csv('dat', cap)
                
            with open('input/params/sim_params.yml') as fp2:
                pars = yaml.load(fp2)   
                
            # Update parameter file for Triumvirate.
            pars['directories']['catalogues'] = 'new_mocks/%s' % cap
            pars['directories']['measurements'] = 'output/pk_sims/%s' % cap
            pars['files']['data_catalogue'] = 'Patchy-Mocks-DR12%s-COMPSAM_V6C_%0.4d.dat.csv' % (cap, i)
            pars['files']['rand_catalogue'] = str(conv['files']['rand_sim'] + '.csv')
            pars['degrees']['ELL'] = ell
            pars['tags']['output'] = '_sim_%s_%s' % (cap, i)
            
            with open('input/params/sim_params.yml', 'w+') as fp2:
                yaml.dump(pars, fp2)
                
            # Calculate power spectrum.  
            pars = ParameterSet(param_filepath='input/params/sim_params.yml')
            calc_pk(pars)
    
    # Calculate covariance.
    cov_matrix = calc_cov(ell, cap, N)
    np.savetxt('output/cov_%s_%s.txt' %(cap, ell), cov_matrix)
    
    return cov_matrix

def plot_corr(cov_matrix, ell, cap):
    '''Plot the correlation matrix.'''

    # Set up figure.
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    # Convert covariance matrix to correlation matrix.
    plt.imshow(cov2corr(cov_matrix))
    plt.colorbar()
    plt.savefig('corr_matrix_%s_%s' %(cap, ell), dpi=1000)

def get_smooth_model(parameters, k, templates):
    ''' 
    Combine a noBAO model with polynomials and linear bias
    to make the fit independent of the broadband. 
    
    Parameters
    ----------
    parameters: p0, p1, etc. 
    k: wavenumber
    templates: models (here, without BAO)
    '''
    polynomials = parameters[1]/k**3 + parameters[2]/k**2 + parameters[3]/k + parameters[4] + parameters[5]*k
    # print(np.shape(parameters[0]*parameters[0]*templates['noBAO'] + polynomials))
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

def get_oscillation(krange, pk_wiggle, pk_no_bao):
    ''' 
    Use the smooth model and chi squared minimisation to
    extract the BAO signal from the power spectrum. 
    
    Parameters
    ----------
    krange: range of wavenumbers in power spectrum
    pk_wiggle: power spectrum model with oscillations
    pk_no_bao: power spectrum model without oscillations
    '''
    cov_inv = np.identity(len(krange))
    start = [1., 0., 0., 0., 0., 0.]
    
    # Chi squared minimisation.
    result = op.minimize(calc_chi2, start, args=([{ 'k': krange, 'pk': pk_wiggle(krange), 'cov_inv': cov_inv }],\
     { 'noBAO': pk_no_bao }, get_smooth_model))
    
    print(result)
    pk = get_smooth_model(result['x'], krange, { 'noBAO': pk_no_bao })
    plt.clf()
    ax = sns.lineplot(x=krange, y=pk*krange)
    plt.savefig('test_smooth', dpi=800)
    # Divide power spectrum with oscillations by smooth model to extract oscillations.
    yolin = pk_wiggle(krange)/get_smooth_model(result['x'], krange, { 'noBAO': pk_no_bao })
    return interp.interp1d(krange, yolin, fill_value='extrapolate')

def within_priors(parameters):
    '''Test for priors.'''
    if len(parameters) > 6. and abs(parameters[6] - 1.) > 0.2:
        return False
    elif len(parameters) > 6. and (parameters[7] < 0. or parameters[7] > 20.):
        return False
    else:
        return True
    
def calc_chi2(parameters, data, templates, func):
    '''Compare the model with the data using chi squared.'''
    if within_priors(parameters):
        chi2 = 0.
        # Loop over all datasets.
        for dataset in data:
            model = func(parameters, dataset['k'], templates)
            diff = (model - dataset['pk'])
            chi2 += np.dot(diff, np.dot(dataset['cov_inv'], diff))
        return chi2
    else:
        return 100000.
    
def get_percentiles(chain, labels):
    '''Calculate constraints and uncertainties from MCMC chain.'''
    per = np.percentile(chain, [50., 15.86555, 84.13445, 2.2775, 97.7225], axis=0)
    per = np.array([per[0], per[0]-per[1], per[2]-per[0], per[0]-per[3], per[4]-per[0]])
    
    for i in range(0, len(per[0])):
        with open('output/percentiles', 'w') as f:
            np.savetxt(f, "%s = %f +%f -%f +%f -%f" % (labels[i], per[0][i], per[1][i], per[2][i], per[3][i], per[4][i]))
    return per

def get_loglike(parameters, data, templates, func):
    '''Get log likelihood.'''
    return -0.5*calc_chi2(parameters, data, templates, func)

def gelman_rubin_convergence(within_chain_var, mean_chain, chain_length):
    ''' 
    Calculate Gelman & Rubin diagnostic.
    1. Remove the first half of the current chains.
    2. Calculate the within chain and between chain variances.
    3. Estimate variance from the within chain and between chain variance.
    4. Calculate the potential scale reduction parameter.
    
    Parameters
    ----------
    within_chain_var: variance for each parameter across all chains
    mean_chain: mean parameter values across all chains
    chain_length: length of each chain
    
    Return
    ------
    scale reduction parameter
    '''
    Nchains = within_chain_var.shape[0]
    dim = within_chain_var.shape[1]
    meanall = np.mean(mean_chain, axis=0)
    print('meanall=', meanall)
    W = np.mean(within_chain_var, axis=0)
    B = np.arange(dim, dtype=float)
    B.fill(0)
    for jj in range(0, Nchains):
        B = B + chain_length*(meanall - mean_chain[jj])**2/(Nchains-1.)
    estvar = (1. - 1./chain_length)*W + B/chain_length
    
    return np.sqrt(estvar/W)

def prep_gelman_rubin(sampler):
    '''
    Prepare chains for testing Gelman-Rubin convergence.
    
    Parameters
    ----------
    sampler: sampler containing information about the MCMC chains.
    
    Return
    ------
    within_chain_var: variance for each parameter across all chains.
    mean_chain: mean parameter values across all chains.
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

def inspect_chain(list_of_samplers, labels=[]):
    ''' Print chain properties '''
    Nchains = len(list_of_samplers)
    dim = list_of_samplers[0].chain.shape[2]
    if not labels:
        # set default labels
        labels = [('para_%i' % i) for i in range(0,dim)]

    mergedsamples = []
    for jj in range(0, Nchains):
        chain_length = list_of_samplers[jj].chain.shape[1]
        mergedsamples.extend(list_of_samplers[jj].chain[:, int(chain_length/2):, :].reshape((-1, dim)))

    # write out chain
    res = open("chain.dat", "w")
    for row in mergedsamples:
        for el in row:
            res.write("%f " % el)
        res.write("\n")
    res.close()

    print("length of merged chain = ", len(mergedsamples))
    try:
        for jj in range(0, Nchains):
            print("Mean acceptance fraction for chain ", jj,": ", np.mean(list_of_samplers[jj].acceptance_fraction))
    except Exception as e:
        print("WARNING: %s" % str(e))
    try:
        for jj in range(0, Nchains):
            print("Autocorrelation time for chain ", jj,": ", list_of_samplers[jj].get_autocorr_time())
    except Exception as e:
        print("WARNING: %s" % str(e))

    fig = corner.corner(mergedsamples, quantiles=[0.16, 0.5, 0.84], plot_density=False,\
        show_titles=True, title_fmt=".3f", labels=labels)
    fig.savefig("corner.png")

    fig, axes = plt.subplots(dim, 1, sharex=True, figsize=(8, 9))
    for i in range(0, dim):
        for jj in range(0, Nchains):
            axes[i].plot(list_of_samplers[jj].chain[:, :, i].T, alpha=0.4)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].set_ylabel(labels[i])
    fig.tight_layout(h_pad=0.0)
    fig.savefig("%stime_series.png")

    try:
        return get_percentiles(mergedsamples, labels)
    except Exception as e:
        print("WARNING: %s" % str(e))
        return None
        
def runMCMC(start, data, templates):
    ''' 
    Perform MCMC.
    
    Parameters
    ----------
    start: starting point for chain parameters
    data: observational data to be fit by MCMC sampling
    templates: model templates being compared to observational data
    '''
    dim = len(start)
    Nchains = 4
    nwalkers = 20
    ichaincheck = 400 # iterations between convergence checks
    minlength = 2000
    epsilon = 0.01

    # Set parameter labels and expected errors.
    labels = ['b', 'A1', 'A2', 'A3', 'A4', 'A5', 'alpha', 'sigmaNL']
    expected_error = [0.1, 1., 1., 1., 1., 1., 0.05, 0.1]
    
    # Initialise walker positions and set up sampler.
    pos = []
    list_of_samplers = []
    for jj in range(0, Nchains):
        pos.append([start + (2.*np.random.random_sample((dim,)) - 1.) * expected_error for i in range(nwalkers)])
        list_of_samplers.append(emcee.EnsembleSampler(nwalkers=nwalkers, ndim=dim, log_prob_fn=get_loglike,\
         args=(data, templates, get_shifted_model)))

    # Start MCMC.
    print("Running MCMC... ")
    within_chain_var = np.zeros((Nchains, dim))
    mean_chain = np.zeros((Nchains, dim))
    scalereduction = np.arange(dim, dtype=float)
    scalereduction.fill(2.)

    itercounter = 0
    chainstep = minlength
    
    # Run sampling and update positions until convergence is reached.
    while any(abs(1. - scalereduction) > epsilon):
        itercounter += chainstep
        for jj in range(0, Nchains):
            for result in list_of_samplers[jj].sample(pos[jj], iterations=chainstep, store=True):
                pos[jj] = result
            # Convergence test on the second half of the current chain (itercounter/2)
            within_chain_var[jj], mean_chain[jj] = prep_gelman_rubin(list_of_samplers[jj])
        scalereduction = gelman_rubin_convergence(within_chain_var, mean_chain, int(itercounter/2))
        print("scalereduction = ", scalereduction)
        chainstep = ichaincheck
    # Investigate the chain and print out some metrics
    return inspect_chain(list_of_samplers, labels) 
    
def read_pk(ell, cap, tag):
    '''
    Read in the power spectrum.
    
    Parameters
    ----------
    ell: multipole degree
    cap: 'NGC' or 'SGC'
    tag: CMASS, LOWZ, etc.
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
    
def pk_class(k):
    # Compute the linear matter power spectrum using CLASS
    # Set the parameters for the CLASS code
    
    class_params = {
        'output': 'mPk'
    }

    # Initialize the CLASS object
    cosmo_class = Class()

    # Set the CLASS parameters
    cosmo_class.set(class_params)
    cosmo_class.compute()

    # Compute the linear matter power spectrum for each k
    Plin = np.array([cosmo_class.pk_lin(ki, 0) for ki in k])
    
    # Plin = k**0.966 * tk_class(k)**2

    # Get the normalization factor
    sigma8 = 0.8102
    norm_factor = sigma8 ** 2 / cosmo_class.sigma8()
    
    print('norm: ', norm_factor)

    # Apply normalization
    Plin *= norm_factor 

    # Clean up and free memory used by CLASS
    cosmo_class.struct_cleanup()
    cosmo_class.empty()
    
    plt.clf()
    ax = sns.lineplot(x=k, y=Plin*k)
    plt.savefig('test_wiggle', dpi=800)

    return Plin  
    
def estimate_dv(ell, tag):
    '''
    Run MCMC to estimate the BAO scale.
    
    Parameters
    ----------
    ell: multipole degree
    tag: CMASS, LOWZ, etc.
    '''
    sgc_data = read_pk(ell, 'SGC', tag)
    ngc_data = read_pk(ell, 'NGC', tag)
    
    pk_sgc = (sgc_data['Re{pk%s_raw}' %ell] - sgc_data['Re{pk%s_shot}' %ell]) 
    pk_ngc = (ngc_data['Re{pk%s_raw}' %ell] - ngc_data['Re{pk%s_shot}' %ell])
    
    cov_sgc = process_sims(ell, 'SGC', 501)
    
    cov_ngc = process_sims(ell, 'NGC', 501)
    cov_sgc_inv = inv(cov_sgc)
    cov_ngc_inv = inv(cov_ngc)

    boss_data = [{'k': sgc_data['k_cen'],\
                  'pk': pk_sgc,\
                  'label': 'BOSS SGC',\
                  'cov': cov_sgc,\
                  'cov_inv': cov_sgc_inv},\
                      
                 {'k': ngc_data['k_cen'],\
                  'pk': pk_ngc,\
                  'label': 'BOSS NGC',\
                  'cov': cov_ngc,\
                  'cov_inv': cov_ngc_inv}]

    pk_no_bao = cosmotools.linearpower.LinearPower(cosmo=cosmo, redshift=0., transfer='NoWiggleEisensteinHu')
    pk_wiggle = cosmotools.linearpower.LinearPower(cosmo=cosmo, redshift=0., transfer='CLASS')

    krange = np.arange(0.005, 0.205, 0.001)

    pk_wiggle2 = pk_class(krange)
    plt.clf()
    ax = sns.lineplot(x=krange, y=pk_wiggle(krange) * krange, label='CLASS Pk', color='r')
    ax = sns.lineplot(x=krange, y=pk_wiggle2 * krange, label='CLASS Pk2', color='magenta')
    ax = sns.lineplot(x=krange, y=pk_sgc * krange, label='data Pk', color='g')
    ax = sns.lineplot(x=krange, y=pk_sgc * krange - pk_wiggle(krange) * krange, label='residuals', color='b')
    ax.set_ylabel('$kP(k)$')
    ax.set_xlabel('$k$')
    plt.axhline(y = 0.5, color = 'black', linestyle = '-') 
    plt.savefig('test_residuals', dpi=800)

    os_model = get_oscillation(krange, pk_wiggle, pk_no_bao)
    start = [2.37, -0.076, 38., -3547., 15760., -22622., 1., 9.41]
    result = op.minimize(calc_chi2, start, args=(boss_data, { 'noBAO': pk_no_bao, 'os_model': os_model }, get_shifted_model))
    # print("result['x'] = ", result['x'])
    
    best_fit_model = get_shifted_model(result["x"], krange, { 'noBAO': pk_no_bao, 'os_model': os_model }) / get_smooth_model(result["x"], krange, { 'noBAO': pk_no_bao })
    boss_data[0]['ratio'] = boss_data[0]['pk']/get_smooth_model(result["x"], boss_data[0]['k'], { 'noBAO': pk_no_bao })
    boss_data[0]['ratio_err'] = np.sqrt(np.diagonal(boss_data[0]['cov']))/get_smooth_model(result["x"], boss_data[0]['k'], { 'noBAO': pk_no_bao })
    boss_data[1]['ratio'] = boss_data[1]['pk']/get_smooth_model(result["x"], boss_data[1]['k'], { 'noBAO': pk_no_bao })
    boss_data[1]['ratio_err'] = np.sqrt(np.diagonal(boss_data[1]['cov']))/get_smooth_model(result["x"], boss_data[1]['k'], { 'noBAO': pk_no_bao })
    
    per = runMCMC(result["x"], boss_data, { 'noBAO': pk_no_bao, 'os_model': os_model })
    
    zmean = 0.50282567
    if per is not None:
        Hz = cosmo.efunc(zmean)*cosmo.H0*cosmo.h
        DC = (1.+zmean)*cosmo.angular_diameter_distance(zmean)/cosmo.h
        c_km_s = (speed_of_light/1000.)
        DVfid = ( DC**2*(zmean*c_km_s/Hz) )**(1./3.)
        DV = per[:,6]*DVfid
        print("DV = %f +%f -%f +%f -%f\n" % (DV[0], DV[1], DV[2], DV[3], DV[4]))
    else:
        print("Problem with MCMC chain")
    return 
        
if __name__ == '__main__':
    # process_data(0, 'SGC', 'LOWZ')
    # process_sims(0, 'NGC', 501)
    # plot_corr(process_sims(0, 'NGC', 501), 0, 'NGC')
    estimate_dv(0, 'CMASS')
