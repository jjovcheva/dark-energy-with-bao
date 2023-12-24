import numpy as np
from classy import Class
from scipy.integrate import quad

available = ['CLASS', 'EisensteinHu', 'NoWiggleEisensteinHu']

KMIN = 1e-8

pars = {'output': 'mPk'}

def scale_independent_growth_factor(cosmo, redshift):

    # Special case for redshift 0 (present day).
    if redshift == 0:
        return 1.0 

    # Convert redshift to scale factor.
    a = 1 / (1 + redshift)

    # Define the integrand for the growth factor.
    def integrand(a_prime):
        H = cosmo.H(a_prime).to_value('km/s/Mpc')  # Convert Hubble parameter to km/s/Mpc.
        return 1 / ((a_prime * H)**3)

    # Perform the integral using quad function.
    integral_result, _ = quad(integrand, a, 1)

    # Calculate the growth factor.
    growth_factor = 5/2 * cosmo.Om0 * cosmo.H0.to_value('km/s/Mpc')**2 * integral_result

    return growth_factor

class CLASS(object):

    def __init__(self, cosmo, redshift):
        
        self.cosmo = cosmo
        cosmo.set(pars)
        cosmo.compute()
        
        # find the low-k amplitude to normalize to 1 as k->0 at z = 0
        self._norm = 1.0; self.redshift = 0
        self._norm = 1. / self(KMIN)

        # the proper redshift
        self.redshift = redshift
        
    def __call__(self, k):
        
        k = np.asarray(k)
        nonzero = k > 0
                
        # Call the function         
        linearP = np.array([self.cosmo.pk_lin(ki, self.redshift) for ki in k[nonzero]]) / self.cosmo.h()**3      
        #linearP = self.cosmo.pk_lin(k[nonzero], np.array([self.redshift])) / self.cosmo.h()**3
        primordialP = (k[nonzero] * self.cosmo.h())**self.cosmo.n_s() # put k into Mpc^{-1}

        Tk = np.ones(nonzero.shape)
        # at k=0, this is 1.0 * D(z), where T(k) = 1.0 at z=0
        Tk[~nonzero] = scale_independent_growth_factor(self.cosmo, self.redshift)

        # fill in all k>0
        Tk[nonzero] = self._norm * (linearP / primordialP)**0.5
        return Tk 

class EisensteinHu(object):

    def __init__(self, cosmo, redshift):

        self.cosmo = cosmo
        self.redshift = redshift

        self.Obh2 = cosmo.Omega_b() * cosmo.h() ** 2
        self.Omh2 = cosmo.Omega0_m() * cosmo.h() ** 2
        self.f_baryon = cosmo.Omega_b() / cosmo.Omega0_m()
        self.theta_cmb = cosmo.T_cmb() / 2.7

        # redshift and wavenumber of equality
        self.z_eq = 2.5e4 * self.Omh2 * self.theta_cmb ** (-4) # this is 1 + z
        self.k_eq = 0.0746 * self.Omh2 * self.theta_cmb ** (-2) # units of 1/Mpc

        # sound horizon and k_silk
        self.z_drag_b1 = 0.313 * self.Omh2 ** -0.419 * (1 + 0.607 * self.Omh2 ** 0.674)
        self.z_drag_b2 = 0.238 * self.Omh2 ** 0.223
        self.z_drag    = 1291 * self.Omh2 ** 0.251 / (1. + 0.659 * self.Omh2 ** 0.828) * \
                           (1. + self.z_drag_b1 * self.Obh2 ** self.z_drag_b2)

        self.r_drag = 31.5 * self.Obh2 * self.theta_cmb ** -4 * (1000. / (1+self.z_drag))
        self.r_eq   = 31.5 * self.Obh2 * self.theta_cmb ** -4 * (1000. / self.z_eq)

        self.sound_horizon = 2. / (3.*self.k_eq) * np.sqrt(6. / self.r_eq) * \
                    np.log((np.sqrt(1 + self.r_drag) + np.sqrt(self.r_drag + self.r_eq)) / (1 + np.sqrt(self.r_eq)) )
        self.k_silk = 1.6 * self.Obh2 ** 0.52 * self.Omh2 ** 0.73 * (1 + (10.4*self.Omh2) ** -0.95)

        # alpha_c
        alpha_c_a1 = (46.9*self.Omh2) ** 0.670 * (1 + (32.1*self.Omh2) ** -0.532)
        alpha_c_a2 = (12.0*self.Omh2) ** 0.424 * (1 + (45.0*self.Omh2) ** -0.582)
        self.alpha_c = alpha_c_a1 ** -self.f_baryon * alpha_c_a2 ** (-self.f_baryon**3)

        # beta_c
        beta_c_b1 = 0.944 / (1 + (458*self.Omh2) ** -0.708)
        beta_c_b2 = 0.395 * self.Omh2 ** -0.0266
        self.beta_c = 1. / (1 + beta_c_b1 * ((1-self.f_baryon) ** beta_c_b2) - 1)

        y = self.z_eq / (1 + self.z_drag)
        alpha_b_G = y * (-6.*np.sqrt(1+y) + (2. + 3.*y) * np.log((np.sqrt(1+y)+1) / (np.sqrt(1+y)-1)))
        self.alpha_b = 2.07 *  self.k_eq * self.sound_horizon * (1+self.r_drag)**-0.75 * alpha_b_G

        self.beta_node = 8.41 * self.Omh2 ** 0.435
        self.beta_b    = 0.5 + self.f_baryon + (3. - 2.*self.f_baryon) * np.sqrt( (17.2*self.Omh2) ** 2 + 1 )

    def __call__(self, k):

        if np.isscalar(k) and k == 0.:
            return 1.0

        k = np.asarray(k)
        # only compute k > 0 modes
        valid = k > 0.

        k = k[valid] * self.cosmo.h() # now in 1/Mpc
        q = k / (13.41*self.k_eq)
        ks = k*self.sound_horizon

        T_c_ln_beta   = np.log(np.e + 1.8*self.beta_c*q)
        T_c_ln_nobeta = np.log(np.e + 1.8*q);
        T_c_C_alpha   = 14.2 / self.alpha_c + 386. / (1 + 69.9 * q ** 1.08)
        T_c_C_noalpha = 14.2 + 386. / (1 + 69.9 * q ** 1.08)

        T_c_f = 1. / (1. + (ks/5.4) ** 4)
        f = lambda a, b : a / (a + b*q**2)
        T_c = T_c_f * f(T_c_ln_beta, T_c_C_noalpha) + (1-T_c_f) * f(T_c_ln_beta, T_c_C_alpha)

        s_tilde = self.sound_horizon * (1 + (self.beta_node/ks)**3) ** (-1./3.)
        ks_tilde = k*s_tilde

        T_b_T0 = f(T_c_ln_nobeta, T_c_C_noalpha)
        T_b_1 = T_b_T0 / (1 + (ks/5.2)**2 )
        T_b_2 = self.alpha_b / (1 + (self.beta_b/ks)**3 ) * np.exp(-(k/self.k_silk) ** 1.4)
        T_b = np.sinc(ks_tilde/np.pi) * (T_b_1 + T_b_2)

        T = np.ones(valid.shape)
        T[valid] = self.f_baryon*T_b + (1-self.f_baryon)*T_c;

        return T * self.cosmo.scale_independent_growth_factor(self.redshift)
     
class NoWiggleEisensteinHu(object):
    '''
    Calculate the matter transfer function without BAO
    using the Eisenstein & Hu (1998) fitting formula.

    Parameters
    ----------
    k: wavenumber array (h/Mpc)

    Return
    ------
    tk_no_bao: matter transfer function without BAO
    '''
    
    def __init__(self, cosmo, redshift):
        # Initialize the CLASS object.
        cosmo = Class()
        
        self.cosmo = cosmo
        self.redshift = redshift
        
        # Set the CLASS parameters.
        cosmo.set(pars)
        cosmo.compute()
                
        # Constants from Eisenstein & Hu (1998)
        
        self.Obh2 = cosmo.Omega_b() * cosmo.h()**2
        self.Omh2 = cosmo.Omega0_m() * cosmo.h()**2
        self.f_baryon = cosmo.Omega_b() / cosmo.Omega0_m()
        self.theta_cmb = cosmo.T_cmb()
        
        self.k_eq = 0.0746 * self.Omh2 * self.theta_cmb ** (-2)
        self.sound_horizon = cosmo.h() * 44.5 * np.log(9.83/self.Omh2) / \
                            np.sqrt(1 + 10 * self.Obh2** 0.75) # in Mpc/h
        self.alpha_gamma = 1 - 0.328 * np.log(431*self.Omh2) * self.f_baryon + \
                            0.38* np.log(22.3*self.Omh2) * self.f_baryon ** 2
        
    def __call__(self, k):
        
        if np.isscalar(k) and k == 0.:
            return 1.0
        
        k = np.asarray(k)
        valid = k > 0.
        
        k = k[valid] * self.cosmo.h() 
        ks = k * self.sound_horizon / self.cosmo.h()
        q = k / (13.41 * self.k_eq)
        
        gamma_eff = self.Omh2 * (self.alpha_gamma + (1 - self.alpha_gamma) / (1 + (0.43*ks)**4))
        q_eff = q * self.Omh2 / gamma_eff
        L = np.log(2 * np.e + 1.8 * q_eff)
        C = 14.2 + 731 / (1 + 62.5 * q_eff)
        
        tk_no_bao = np.ones(valid.shape)
        tk_no_bao[valid] = L / (L + C * q_eff**2)
        
        return tk_no_bao * self.cosmo.scale_independent_growth_factor(self.redshift)
    
    