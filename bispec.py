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
from scipy.constants import speed_of_light
from numpy.linalg import inv

from triumvirate.catalogue import ParticleCatalogue
from triumvirate.logger import setup_logger
from triumvirate.parameters import ParameterSet
from triumvirate.parameters import fetch_paramset_template
from triumvirate.threept import compute_bispec, compute_bispec_in_gpp_box

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Serif"],
})

yaml = ruamel.yaml.YAML()

cat_dir = 'input/catalogues'
conv_file = 'cat_conv.yml'

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
    
    cat_data = fits.open(os.path.join('input/catalogues', 
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
    wc = np.array(cat_ran['WEIGHT_FKP'], dtype=np.float64)
    nz = np.array(cat_ran['NZ'], dtype=np.float64)
    
    return cat_ran, nz, ws, wc

def calc_comoving_distance(redshift):
    '''Calculate the comoving distance in Mpc given a redshift.'''
    return cosmo.comoving_distance(redshift).to('Mpc').value

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

def calc_bk(pars):
    '''Calculate the bispectrum.'''
    
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
    measurements = compute_bispec(
        cat_data, cat_rand, paramset=pars, save='.txt'
        )
    
def process_data(ell1, ell2, ELL, cap, tag):
    '''
    Calculate bispectra for galaxy data.
    
    Parameters
    ----------
    ells: multipole degrees
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
    
    pars['degrees']['ell1'] = ell1
    pars['degrees']['ell2'] = ell2
    pars['degrees']['ELL'] = ELL
    pars['tags']['output'] = '_%s_%s' % (cap, tag)
    
    with open('input/params/bk_params.yml', 'w+') as fp2:
        yaml.dump(pars, fp2)
    
# Calculate bispectrum.
pars = ParameterSet(param_filepath='input/params/bk_params.yml')
calc_bk(pars)