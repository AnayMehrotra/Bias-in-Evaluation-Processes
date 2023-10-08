import math
import numpy as np
from helpers import *
from tqdm.notebook import tqdm


class Bias_Framework:
    '''
        This is an implementation of the framework in the paper
        
        The parameters of this class describe the parameters (alpha, tau, error
        function, ...) describe the parameters of the optimization problem in the paper

        The main function is `get_biased_utility` which given an evaluation `v`
        outputs a biased evaluation `hat{v}` sampled from the distribution described in the paper 
        TODO
    '''
    
    def __init__(self, alpha, tau, get_lagrangian_integral = None, f_0 = None, OMEGA = 'R', required_gamma_precision=None) -> None: 
        '''
            Initialize the framework:
            args:
                alpha:             scalar value larger than 0
                tau:               scalar value (either positive or negative)
                TODO
        '''
        self.alpha = alpha
        self.tau = tau
        self.required_gamma_precision = required_gamma_precision
        
        self.gamma = -1
        self.get_lagrangian_integral = get_lagrangian_integral
        
        self.f_0 = f_0 if f_0 != None else lambda x: 1
        self.f_ = lambda hat_v, gamma_, alpha: EXP(- self.get_lagrangian_integral(hat_v, alpha) / gamma_)

        if type(OMEGA) == type('a'):
            assert(OMEGA in ['R', 'R>=0'])
        
        self.OMEGA = OMEGA

        # minimum and maximum range of interval over which utilities are supported
        if type(self.OMEGA) == type('str') and self.OMEGA == 'R':
            self.OMEGA_MIN = -10
            self.OMEGA_MAX = 10
            self.discrete = False
        elif type(self.OMEGA) == type('str') and self.OMEGA == 'R>=0':
            self.OMEGA_MIN = 0
            self.OMEGA_MAX = 10
            self.discrete = False
        elif type(self.OMEGA) == type((1,2)):  
            self.OMEGA_MIN = OMEGA[0]
            self.OMEGA_MAX = OMEGA[1]
            self.discrete = False
        else:
            # discrete set
            self.OMEGA_MIN = min(self.OMEGA)
            self.OMEGA_MAX = max(self.OMEGA)
            self.discrete = True

    def set_alpha(self, alpha_):
        self.alpha = alpha_
    
    def set_tau(self, tau_):
        self.tau = tau_

    def set_get_lagrangian_integral(self, get_lagrangian_integral_):
        self.get_lagrangian_integral = get_lagrangian_integral_
        self.f_ = lambda hat_v, gamma_, alpha: EXP(- self.get_lagrangian_integral(hat_v, alpha) / gamma_)

    def set_omega(self, OMEGA):
        assert(OMEGA in ['R', 'R>=0'] or type(OMEGA) == type((1,2)))
        self.OMEGA = OMEGA

        # minimum and maximum range of interval over which utilities are supported
        if type(self.OMEGA) == type('str') and self.OMEGA == 'R':
            self.OMEGA_MIN = -10
            self.OMEGA_MAX = 10
            self.discrete = False
        elif type(self.OMEGA) == type('str') and self.OMEGA == 'R>=0':
            self.OMEGA_MIN = 0
            self.OMEGA_MAX = 10
            self.discrete = False
        elif type(self.OMEGA) == type((1,2)): 
            self.OMEGA_MIN = OMEGA[0]
            self.OMEGA_MAX = OMEGA[1]
            self.discrete = False
        else:
            # discrete set
            self.OMEGA_MIN = min(self.OMEGA)
            self.OMEGA_MAX = max(self.OMEGA)
            self.discrete = True

    def set_f0(self, f_0):
        self.f_0 = f_0

    def get_biased_distribution(self, verbose=False):
        '''
            Solves the optimization framework in the paper
            args:
                pdf_v:      A function giving density of v 
            output: 
                    f:      A function giving density of \hat{v}
        '''
        # lagrangian variable, gamma, is chosen to ensure the entropy of f is exactly - self.tau
        self.gamma = self.find_gamma(verbose=verbose)
        if verbose: print(f'\t gamma={self.gamma}')

        # const is chosen to ensure that integral of f(x) is 1 
        I = lambda x: ONE * self.f_0(x) * self.f_(x, self.gamma, self.alpha)
        const = 1 / integrate(I, MIN=self.OMEGA_MIN, MAX=self.OMEGA_MAX, is_discrete=self.discrete, Omega=self.OMEGA) # normalizing constant
        return lambda x:  I(x) * const

    def find_gamma(self, l=1e-10, u=int(5*1e4), verbose = False):
        '''
            Binary search to find the right lagrangian parameter 
            Assumes that the entropy of the output distribution is monotone

            in the Lagrangian parameter; this appears to be true in all cases
            but needs to be proved (TODO)
            
            Args:
                l:              Lower bound on Lagrangian parameter
                u:              Upper bound on Lagrangian parameter
        '''
        TOL = 1e-15
        kl_div = 1e10

        if USE_LONG_MATH:
            l *= ONE
            u *= ONE
            TOL *= ONE
            kl_div *= ONE

        def not_accurate_enough(kl_div, l, u):
            if self.required_gamma_precision is not None:
                return abs(kl_div - self.tau) > 1e-3 or l < u - self.required_gamma_precision
            return abs(kl_div - self.tau) > 1e-3 and l < u - TOL

        while not_accurate_enough(kl_div, l, u):
            m = (l + u) / 2 
            
            if verbose: print('\t getting KL div')
            kl_div = self.get_kl_div(gamma_=m, verbose=verbose) 

            if verbose: print(f'\t (l,u,m)\t=\t{(l,u,m)}', flush=True)
            if verbose: print(f'\t kl_div\t=\t{kl_div}', flush=True)
            if verbose: print(f'\t g(tau)\t=\t{self.tau}', flush=True)
    
            if kl_div > self.tau:
                l = m
            else:
                u = m 
        return l       
    
    def get_kl_div(self, gamma_, verbose=False):
        assert(self.f_ is not None)

        # Normalize f_
        I = lambda x: ONE * self.f_0(x) * self.f_(hat_v=x, gamma_=gamma_, alpha=self.alpha) 
        norm_ = integrate(I, MIN=self.OMEGA_MIN, MAX=self.OMEGA_MAX, is_discrete=self.discrete, Omega=self.OMEGA, verbose=verbose)
        if verbose: print(f'\t\tnorm_={norm_}, discrete={self.discrete}, len(self.OMEGA)={len(self.OMEGA)}')
        
        kl_div_ = lambda x: I(x) / norm_ * LOG( I(x) / norm_ / self.f_0(x)) if I(x) > 0 else 0

        kl_div_val = integrate(kl_div_, MIN=self.OMEGA_MIN, MAX=self.OMEGA_MAX, is_discrete=self.discrete, Omega=self.OMEGA)
        if verbose: print(f'kl_div_val={kl_div_val}')
        
        if ISNAN(kl_div_val): 
            kl_div_val = integrate(kl_div_, MIN=self.OMEGA_MIN, MAX=self.OMEGA_MAX, is_discrete=self.discrete, Omega=self.OMEGA, verbose=verbose)
            assert(False)

        return kl_div_val