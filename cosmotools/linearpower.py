import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from .tk_models import CLASS, NoWiggleEisensteinHu, EisensteinHu
from classy import Class
import mcfit
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.integrate import simps

pars = {'output': 'mPk'}

class LinearPower(object):
    
    def __init__(self, cosmo, redshift, transfer='CLASS'):
        
        # Initialize the CLASS object.
        cosmo = Class()

        # Set the CLASS parameters.
        cosmo.set(pars)
        cosmo.compute()
        
        # Set the CLASS cosmology as the cosmology object.
        self.cosmo = cosmo

        # Set sigma8 to the cosmology value.
        self._sigma8 = cosmo.sigma8

        # Set up the transfers.
        if transfer not in ['CLASS', 'NoWiggleEisensteinHu', 'EisensteinHu']: 
            raise ValueError("'transfer' should be one of CLASS, NoWiggleEisensteinHu, EisensteinHu")
        self.transfer = transfer

        # Initialise CLASS for transfers.
        c = Class()
        c.set(pars)
        c.compute()

        # Use the correct class names based on the available classes in tk_models.
        if transfer == 'CLASS':
            transfer_class = CLASS
        elif transfer == 'NoWiggleEisensteinHu':
            transfer_class = NoWiggleEisensteinHu
        elif transfer == 'EisensteinHu':
            transfer_class = EisensteinHu
        else:
            raise ValueError("Unexpected 'transfer' value")

        self._transfer = transfer_class(c, redshift)
        
        # Fall back to the analytic model when out of range.
        self._fallback = EisensteinHu(c, redshift)

        # Normalize to proper sigma8.
        self._norm = 1.
        self.redshift = 0;
        
        self._norm = (self._sigma8() / self.sigma_r(8.))**2  # sigma_r(z=0, r=8)

        # Set redshift.
        self.redshift = redshift

        # Store meta-data.
        self._attrs = {}
        self._attrs['transfer'] = transfer

    @property
    def attrs(self):
        """
        The meta-data dictionary
        """
        self._attrs['redshift'] = self.redshift
        self._attrs['sigma8'] = self.sigma8
        return self._attrs

    @property
    def redshift(self):
        """
        The redshift of the power spectrum
        """
        return self._z

    @redshift.setter
    def redshift(self, value):
        self._z = value
        self._transfer.redshift = value
        self._fallback.redshift = value
        
    @property
    def sigma8(self):
        """
        The present day value of sigma_r(r=8 Mpc/h), used to normalize
        the power spectrum, which is proportional to the square of this value.

        The power spectrum can re-normalized by setting a different
        value for this parameter
        """
        return self._sigma8

    @sigma8.setter
    def sigma8(self, value):
        """
        Set the sigma8 value and normalize the power spectrum to the new value
        """
        # Re-scale the normalization.
        self._norm *= (value / self._sigma8)**2

        # Update to this sigma8.
        self._sigma8 = value

    def __call__(self, k):
        """
        Return the linear power spectrum in units of
        :math: h^{-3} \mathrm{Mpc}^3 at the redshift specified by
        :attr:'redshift'.

        The transfer function used to evaluate the power spectrum is
        specified by the 'transfer' attribute.

        Parameters
        ----------
        k : float, array_like
            the wavenumber in units of :math: h Mpc^{-1}

        Return
        ------
        Pk : float, array_like
            the linear power spectrum evaluated at k in units of
            :math: h^{-3} \mathrm{Mpc}^3
        """
        if self.transfer != "CLASS":
            Pk = k**self.cosmo.n_s() * self._transfer(k)**2
        else:
            k = np.asarray(k)
            kmax = 1.
            inrange = k < 0.99999*kmax # prevents rounding errors

            # the return array (could be scalar array)
            Pk = np.zeros_like(k)

            # k values in and out of valid range
            k_in = k[inrange]; k_out = k[~inrange]

            # use CLASS in range
            #Pk[inrange] = k_in**self.cosmo.n_s() * self._transfer(k_in)**2
            Pk[inrange] = np.array([self.cosmo.pk_lin(ki, 0) for ki in k_in]) 

            # use Eisentein-Hu out of range
            if len(k_out):
                analytic_Tk = self._fallback(k_out)
                analytic_Tk *= self._transfer(kmax)/ self._fallback(kmax)
                Pk[~inrange] = k_out**self.cosmo.n_s() * analytic_Tk**2
                
        return self._norm * Pk 

    def sigma_r(self, r, kmin=1e-5, kmax=1e0):

        k = np.logspace(np.log10(kmin), np.log10(kmax), 1024)
        Plin = self(k)
        
        # Get variance from top-hat window function.
        R, sigmasq = mcfit.TophatVar(k, lowring=True)(Plin, extrap=True)

        return spline(R, sigmasq)(r)**0.5
    
    def sigma_r_opt2(self, filter_radius=8., kmin=1e-5, kmax=1e0):
        # This function performs the same calculation as sigma_r. 
        # It makes little difference which one is used.

        k = np.logspace(np.log10(kmin), np.log10(kmax), 1024)
        
        # Get the linear power spectrum at a specific redshift.
        Plin = self(k)
        
        # Top-hat window function.
        def top_hat_window(x):
            return 3 * (np.sin(x) - x * np.cos(x)) / x**3

        # Calculate the integrand for the variance.
        integrand = k**2 * Plin * top_hat_window(k * filter_radius)**2 / (2 * np.pi**2)

        # Integrate to get the variance.
        sigma_r_squared = simps(integrand, k)
        sigma_r = np.sqrt(sigma_r_squared)

        return sigma_r
