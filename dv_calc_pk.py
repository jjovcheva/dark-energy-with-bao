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
    # print(np.shape(parameters[0]*parameters[0]*templates['noBAO'](k) + polynomials))
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
            #print(model)
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
    within_chain_var: variance for each parameter across all chains.
    mean_chain: mean parameter values across all chains.
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

cosmo = cosmology.setCosmology('WMAP9')
 
def pk_wiggle(krange):
    norm = cosmo.sigma(R=8, z=0.0, ps_args={'model': 'camb'})**2
    
    return cosmo.matterPowerSpectrum(krange, z = 0.0, model='camb') / norm

def pk_no_bao(krange):
    norm = cosmo.sigma(R=8, z=0.0, ps_args={'model': 'eisenstein98_zb'})**2
    
    return cosmo.matterPowerSpectrum(krange, z = 0.0, model='eisenstein98_zb') / norm
            
def cmp_ratio_with_error(pk1, pk2, models=[]):
    ''' Compare two power spectra including uncertainties. '''
    plt.clf()
    for model in models:
        plt.plot(model['k'], model['pk'])
    plt.errorbar(pk1['k'], pk1['ratio'], yerr=pk1['ratio_err'], marker='.',\
     linestyle = 'None', label=pk1['label'])
    plt.errorbar(pk2['k'], pk2['ratio'], yerr=pk2['ratio_err'], marker='.',\
     linestyle = 'None', label=pk2['label'])
    plt.axhline(y=1., color='black', linestyle='--')
    plt.legend(loc=0)
    plt.xlabel("k [$h$Mpc$^{-1}$]")
    plt.ylabel("$P(k)/P^{noBAO}(k)$")
    plt.xlim(0.01, 0.3)
    plt.savefig('cmp_ratio')
    return



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
    
    cov_sgc = calc_cov(ell, 'SGC', 501)
    cov_ngc = calc_cov(ell, 'NGC', 501)
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

    krange = sgc_data['k_cen']

    os_model = get_oscillation(krange, pk_wiggle, pk_no_bao)
    start = [2.37, -0.076, 38., -3547., 15760., -22622., 1., 9.41]

    result = op.minimize(calc_chi2, start, args=(boss_data, {'noBAO': pk_no_bao, 'os_model': os_model}, get_shifted_model))
    print("result['x'] = ", result['x'])
    
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
    estimate_dv(0, 'CMASS')
