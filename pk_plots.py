from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import corner
import seaborn as sns
import os
from statistics import mean
from statsmodels.stats.moment_helpers import cov2corr

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Serif"],
})

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

def plot_pk(ell, tag):
    '''
    Plot the power spectra.
    
    Parameters
    ----------
    ell: multipole degree
    tag: CMASS, LOWZ, etc.
    '''
    
    plt.clf()
    sns.set_palette(sns.color_palette('hls', 2))
    
    for cap in ['SGC', 'NGC']:
        pk = read_pk(ell, cap, tag)  
        
        if cap == 'NGC':
            pk_tot = pk['Re{pk%s_raw}' %ell] - pk['Re{pk%s_shot}' %ell]
            
            ax = sns.lineplot(
                x=pk['k_cen'], y=pk['k_cen'] * pk_tot, 
                label='NGC', color='g')
            
        elif cap == 'SGC':
            pk_tot = pk['Re{pk%s_raw}' %ell] - pk['Re{pk%s_shot}' %ell]
            
            ax = sns.lineplot(
                x=pk['k_cen'], y=pk['k_cen'] * pk_tot, 
                label='SGC', color='r')
    
    ax.set_xlabel("$k$ [$h$Mpc$^{-1}$]")
    ax.set_ylabel("$kP_%s(k)$ [$h^{-2}$Mpc$^2$]" %ell)
    #ax.set_xlim(0.01, 0.21)
    # ax.set_ylim(-5, 30)
    plt.rcParams['figure.dpi'] = 1000
    plt.savefig('output/plots/pk%s_%s' %(ell, tag), dpi=1000)
    
plot_pk('0', 'LOWZ')

def plot_corr(cov_matrix, ell):
    '''Plot the correlation matrix.'''

    # Set up figure.
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    # Convert covariance matrix to correlation matrix.
    plt.imshow(cov2corr(cov_matrix), cmap='viridis')
    plt.colorbar()
    plt.savefig('corr_matrix_%s' %ell, dpi=1000)