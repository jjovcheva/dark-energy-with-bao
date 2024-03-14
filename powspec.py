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
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from colossus.cosmology import cosmology

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

zmin = 0.5
zmax = 0.75

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
    cat_data = cat_data[(cat_data['Z'] > zmin) & (cat_data['Z'] < zmax)]
        
    # Create combined systematic weight column.
    weight = cat_data['WEIGHT_SYSTOT'] * \
            (cat_data['WEIGHT_NOZ'] + cat_data['WEIGHT_CP'] - 1.0)
    ws = np.array(weight, dtype=np.float64)
    wc = np.array(cat_data['WEIGHT_FKP'], dtype=np.float64)
    nz = np.array(cat_data['NZ'], dtype=np.float64)
    
    zmean = np.average(cat_data['Z'], weights=(weight*cat_data['WEIGHT_FKP']))

    print(zmean)
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
    cat_ran = cat_ran[(cat_ran['Z'] > zmin) & (cat_ran['Z'] < zmax)]
    
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
        sep='\s+',
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
       
    if not os.path.isfile(
        os.path.join(conv['directories']['new_cats'], conv['files']['data_catalogue'])):
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
        print('Calculating %s... ' % pk_file)

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

if __name__ == '__main__':
    # process_data(0, 'SGC', 'CMASSLOWZTOT')
    process_sims(0, 'SGC', 510)
    # plot_corr(process_sims(0, 'NGC', 101), 0, 'NGC')
