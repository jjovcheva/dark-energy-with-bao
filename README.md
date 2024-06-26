# Constraining Dark Energy with Baryon Acoustic Oscillations

## Calculations
`powspec.py` converts the catalogues (in either FITS or CSV format) to a format that is compatible with `Triumvirate`, calculates the power spectra for data and mocks, and uses the mocks to obtain a covariance matrix. The filenames and input/output directories for catalogue conversion should be specified in `cat_conv.yml`. The input (filename, multipole degree, desired statistic, etc) is passed to `Triumvirate` from `pk_params.yml` for data, and `sim_params.yml` for mocks. `dv_calc.py` fits 2 model templates to the data to isolate the BAO signal: a power spectrum _with_ BAO, calculated using the Boltzmann code `CAMB`, and a power spectrum _without_ BAO based on the Eisenstein & Hu (1998) analytic model. `dv_calc.py` then uses Markov Chain Monte Carlo (MCMC) methods to constrain the spherically averaged distance, $D_v$. 

## Acknowledgments
Many thanks to Dr. Florian Beutler (https://mystatisticsblog.blogspot.com), for the project pipeline and the original code upon which this project was based (without which this would have been a significantly longer and more arduous endeavour). Additional thanks to Dr. Mike Wang, for his `Triumvirate` package (https://github.com/MikeSWang/Triumvirate) and extensive assistance with this tool.
