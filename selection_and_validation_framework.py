import math
import copy 
import numpy as np
from helpers import *
import itertools
from tqdm.notebook import tqdm

from framework import Bias_Framework


##############################################################################################################
#########################################         Misc helpers         #######################################
##############################################################################################################

def get_constrained_solution(hat_v1, hat_v2, l1, l2, k):
    ##########
    # S_CONS: Constrainted algorithm
    ##########
    
    # Satisfy lower bounds for group 1
    s_cons = list(hat_v1.argsort()[::-1][:l1])  
    # Satisfy lower bounds for group 2
    s_cons += list(len(hat_v1) + hat_v2.argsort()[::-1][:l2])  

    while len(s_cons) < k:
        if hat_v1[hat_v1.argsort()[::-1][l1]] > hat_v2[hat_v2.argsort()[::-1][l2]]:
            s_cons.append(hat_v1.argsort()[::-1][l1])
            l1 += 1
        else:
            s_cons.append(len(hat_v1) + hat_v2.argsort()[::-1][l2])
            l2 += 1

    return s_cons

def get_shift(y, z, s):
    for i in range(len(y)):
        if i+s >= 0 and i+s < len(y):
            z[i] = y[i + s]
        else:
            z[i] = 0

    return z

def get_mean(y, Omega):
    mean = 0
    for i, v in enumerate(Omega):
        mean += v * y[i]  
    
    return mean 

def get_variance(y, mean, Omega):
    var = 0
    for i, v in enumerate(Omega):
        var += y[i] * (v - mean)**2
    
    return var 


##############################################################################################################
###################################         Selection framework(s)         ###################################
##############################################################################################################
def run_simulation(n1 = 1000, n2 = 1000, ITER = 500, alpha_list = [1], tau_list = [-2], shift_list = [],
                    f_0 = lebesgue_measure, Omega = [], pdf_latent = [0], plot = False, show_legend = True, save_fig = False,
                    Delta_alpha=1e-2, Delta_tau=1e-2, verbose=False, err_func = err_func_l2):
    
    k_list = np.array( range(50, 1000, 10) ) 

    u_cons_er = {}
    u_cons_pr = {}
    u_opt = {}
    u_uncons_alpha = {}
    u_uncons_tau = {}
    u_uncons = {}
    u_list = [u_cons_er, u_cons_pr, u_uncons_alpha, u_uncons_tau, u_opt, u_uncons]

    u_cons_er_err = {}
    u_cons_pr_err = {}
    u_opt_err = {}
    u_uncons_alpha_err = {}
    u_uncons_tau_err = {}
    u_uncons_err = {}
    u_list_err = [u_cons_er_err, u_cons_pr_err, u_uncons_alpha_err, u_uncons_tau_err, u_opt_err, u_uncons_err]

    integrator = get_integrator(err_func, omega=Omega, pdf_v=pdf_latent)
    pbar =  zip(alpha_list, tau_list, shift_list)
    
    for alpha, tau, shift in pbar:
        key = (alpha, tau)
        for u, u_err in zip(u_list, u_list_err):
            u[key] = [] 
            u_err[key] = [] 
        
        ########################################################################
        ## Compute biased densities
        ########################################################################
        if verbose: print('initializing', flush=True)
        framework = Bias_Framework(alpha=alpha, tau=tau, get_lagrangian_integral=integrator, f_0=f_0, OMEGA=Omega)

        if verbose: print('querying...', flush=True)
        get_f = framework.get_biased_distribution()
        pdf_biased = np.array([get_f(v) for i, v in enumerate(Omega)])
        pdf_biased /= sum(pdf_biased)

        if verbose: print('querying shifted alpha...', flush=True)
        framework.set_alpha(min(alpha + Delta_alpha, 1) if alpha < 1 else max(1, alpha - Delta_alpha))
        get_f = framework.get_biased_distribution()
        pdf_biased_alpha = np.array([get_f(v) for i, v in enumerate(Omega)])
        pdf_biased_alpha /= sum(pdf_biased_alpha)
        framework.set_alpha(alpha)

        if verbose: print('querying shifted tau...', flush=True)
        framework.set_tau(tau - Delta_tau)
        get_f = framework.get_biased_distribution()
        pdf_biased_tau = np.array([get_f(v) for i, v in enumerate(Omega)])
        pdf_biased_tau /= sum(pdf_biased_tau)
        framework.set_tau(tau)
        ########################################################################

        tmp = np.zeros_like(pdf_biased)
        get_shift(pdf_biased, tmp, shift) 
        pdf_biased = tmp
        
        tmp = np.zeros_like(pdf_biased_alpha)
        get_shift(pdf_biased_alpha, tmp, shift) 
        pdf_biased_alpha = tmp

        tmp = np.zeros_like(pdf_biased_tau)
        get_shift(pdf_biased_tau, tmp, shift) 
        pdf_biased_tau = tmp

        if plot:
            plt.figure().clear()
            plt.title(f'$n_1={n1}$, $n_2={n2}$, ITER={ITER}, ' 
                      + '$\\Delta_{\\alpha}=' + f'{np.round(Delta_alpha,3)}$, '
                      + '$\\Delta_{\\tau}=' + f'{np.round(Delta_tau,3)}$, '
                      + f'$\\alpha={np.round(alpha, 3)}$, $\\tau={np.round(tau, 3)}$', fontsize=20)
        
            plt.plot(Omega, pdf_latent, label='Latent utilities', linewidth=6)
            plt.plot(Omega, pdf_biased, label='Biased utilities', linewidth=6, linestyle='dashed')
            plt.plot(Omega, pdf_biased_alpha, label=f'Utilities w/ reduced skew ($|\\alpha-1|$)', linewidth=6, linestyle='dashed')
            plt.plot(Omega, pdf_biased_tau, label=f'Utilities w/ decreased $\\tau$', linewidth=6, linestyle='dotted')
            plt.tick_params(axis='both', which='major', labelsize=22)
            if show_legend: plt.legend(fontsize=20)
            plt.show()
            plt.close() 

        if verbose: print('starting simulation', flush=True)
        for k in tqdm( k_list ):
            u_list_ = [[] for i in range(6)] 

            for _ in range(ITER):    
                
                ########################################################################
                ## Sample utilities
                ########################################################################
                v1 = np.random.choice(size=n1, a=Omega, p=pdf_latent)
                hat_v1 = v1

                # Sample biased utilities and maintain a coupling between latent and (different) biased utilities 
                rand_bits = np.random.random(size = n1 + n2)
                v2 = Omega[ np.searchsorted( np.cumsum(pdf_latent), rand_bits) ]
                hat_v2 = Omega[ np.searchsorted( np.cumsum(pdf_biased), rand_bits) - 1 ]
                hat_v2_alpha = Omega[ np.searchsorted( np.cumsum(pdf_biased_alpha), rand_bits) - 1 ]
                hat_v2_tau = Omega[ np.searchsorted( np.cumsum(pdf_biased_tau), rand_bits) - 1 ]

                # Merge utilities
                v = np.array( list(v1) + list(v2) )
                hat_v = np.array( list(hat_v1) + list(hat_v2) )
                hat_v_alpha = np.array( list(hat_v1) + list(hat_v2_alpha) )
                hat_v_tau = np.array( list(hat_v1) + list(hat_v2_tau) )
                ########################################################################

                ########################################################################
                ## Compute outputs of different selection algorithms
                ########################################################################
                s_opt = v.argsort()[::-1][:k] 
                s_uncons = hat_v.argsort()[::-1][:k]
                s_uncons_alpha = hat_v_alpha.argsort()[::-1][:k]
                s_uncons_tau = hat_v_tau.argsort()[::-1][:k]
                s_cons_er = get_constrained_solution(hat_v1, hat_v2, k//2, k//2, k)

                l1 = int(k * n1 / (n1 + n2))
                s_cons_pr = get_constrained_solution(hat_v1, hat_v2, l1, k - l1, k)

                # Compute utilities of selection
                s_list = [s_cons_er, s_cons_pr, s_uncons_alpha, s_uncons_tau, s_opt, s_uncons]

                for i in range(len(u_list_)):
                    assert(len(s_uncons) == len(s_list[i]))
                    u_list_[i].append( np.sum(v[s_list[i]]) )


            for i in range(len(u_list)):
                u_list[i][key].append( np.mean([u_list_[i][j] / u_list_[-1][j] for j in range(ITER)]) )
                u_list_err[i][key].append( np.std([u_list_[i][j] / u_list_[-1][j] for j in range(ITER)]) / np.sqrt(ITER) )

            u_uncons[key].append( np.mean(u_list_[-1]) )
            u_uncons_err[key].append( np.std(u_list_[-1]) / np.sqrt(ITER) )
        
        
        for u, u_err in zip(u_list, u_list_err):
            u[key] = np.array(u[key])
            u_err[key] = np.array(u_err[key])

        plt.figure().clear()
         
        
        ########################################
        # # COMPARISON OF INTERVENTIONS
        ########################################
        plt.title(f'$n_1={n1}$, $n_2={n2}$, ITER={ITER}, ' 
                      + '$\\Delta_{\\alpha}=' + f'{np.round(Delta_alpha,3)}$, '
                      + '$\\Delta_{\\tau}=' + f'{np.round(Delta_tau,3)}$, '
                      + f'$\\alpha={np.round(alpha, 3)}$, $\\tau={np.round(tau, 3)}$', fontsize=20)

        plt.errorbar(k_list, u_uncons_alpha[key], yerr=u_uncons_alpha_err[key], linewidth = 6, label = f'Reduce skew ($|\\alpha-1|$)')
        plt.errorbar(k_list, u_uncons_tau[key], yerr=u_uncons_tau_err[key], linewidth = 6, label = f'Decrease $\\tau$', linestyle='dashed')
        plt.errorbar(k_list + 5, u_cons_er[key], yerr=u_cons_er_err[key], linewidth = 6, label = f'Equal representation' , linestyle='dashed')
        plt.errorbar(k_list, u_cons_pr[key], yerr=u_cons_pr_err[key], linewidth = 6, label = f'Proportional representation', linestyle='dotted')

        if show_legend: plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=4, fontsize=30)
        
        plt.ylim(0.9, 1.4)
        
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.ylabel("Relative increase in utility", fontsize=34)
        plt.xlabel('$k$', fontsize=34)

        if save_fig:    
            pdf_savefig()
        else:   
            plt.show()
        plt.close()

    return 


 ##############################################################################################################

##############################################################################################################
##############################################################################################################

def run_case_study(n1 = 1000, n2 = 1000, k = 1000, ITER = 500, alpha = None, tau = None, shift = None,
                    Omega = [], pdf_latent = [0], k_non_gen = 1000,
                    show_legend = True, save_fig = False,
                    delta_alpha_list=[1e-2], delta_tau_list=[1e-2], verbose=False):
    
    f_0 = get_pdf_uniform_omega(Omega)
    err_func = err_func_l2

    u_cons_er = {}
    u_cons_aq = {} # Actual quota
    u_cons_pr = {}
    u_opt = {}
    u_uncons_alpha = {}
    u_uncons_tau = {}
    u_uncons = {}
    u_list = [u_cons_er, u_cons_aq, u_cons_pr, u_uncons_alpha, u_uncons_tau, u_opt, u_uncons]

    u_cons_er_err = {}
    u_cons_aq_err = {}
    u_cons_pr_err = {}
    u_opt_err = {}
    u_uncons_alpha_err = {}
    u_uncons_tau_err = {}
    u_uncons_err = {}
    u_list_err = [u_cons_er_err, u_cons_aq_err, u_cons_pr_err, u_uncons_alpha_err, u_uncons_tau_err, u_opt_err, u_uncons_err]

    def get_normalized_and_shifted(y, shift):
        y /= np.sum(y)
        tmp = np.zeros_like(y)
        get_shift(y, tmp, shift) 
        y = tmp
        return y


    get_shifted_alpha = lambda alpha, delta: max(min(alpha + alpha *  delta, 1), 0) if alpha < 1 else max(1, alpha - alpha * delta)
    get_shifted_tau = lambda tau, delta: tau - tau * delta
    
    ########################################################################
    ## Compute biased densities
    ########################################################################
    if verbose: print('initializing', flush=True)
    integrator = get_integrator(err_func, omega=Omega, pdf_v=pdf_latent) 
    framework = Bias_Framework(alpha=alpha, tau=tau, get_lagrangian_integral=integrator, f_0=f_0, OMEGA=Omega)

    if verbose: print('querying...', flush=True)
    get_f = framework.get_biased_distribution()
    pdf_biased = np.array([get_f(v) for i, v in enumerate(Omega)])
    pdf_biased = get_normalized_and_shifted(pdf_biased, shift)
    ########################################################################

    for delta_alpha in tqdm(delta_alpha_list):
        key = (delta_alpha)
        for u, u_err in zip(u_list, u_list_err):
            u[key] = [] 
            u_err[key] = [] 

        if verbose: print('querying shifted alpha...', flush=True)
        framework.set_alpha(get_shifted_alpha(alpha, delta_alpha))
        get_f = framework.get_biased_distribution()
        framework.set_alpha(alpha)

        pdf_biased_alpha = np.array([get_f(v) for v in Omega])
        pdf_biased_alpha = get_normalized_and_shifted(pdf_biased_alpha, shift)
            
        u_list_ = [[] for i in range(len(u_list))] 

        for _ in range(ITER):    
            
            ########################################################################
            ## Sample utilities
            ########################################################################
            v1 = np.random.choice(size=n1, a=Omega, p=pdf_latent)
            hat_v1 = v1

            # Sample biased utilities and maintain a coupling between latent and (different) biased utilities 
            rand_bits = np.random.random(size = n1 + n2)
            v2 = Omega[ np.searchsorted( np.cumsum(pdf_latent), rand_bits) ]
            hat_v2 = Omega[ np.searchsorted( np.cumsum(pdf_biased), rand_bits) - 1 ]
            hat_v2_alpha = Omega[ np.searchsorted( np.cumsum(pdf_biased_alpha), rand_bits) - 1 ]

            # Merge utilities
            v = np.array( list(v1) + list(v2) )
            hat_v = np.array( list(hat_v1) + list(hat_v2) )
            hat_v_alpha = np.array( list(hat_v1) + list(hat_v2_alpha) )
            ########################################################################

            ########################################################################
            ## Compute outputs of different selection algorithms
            ########################################################################
            s_opt = v.argsort()[::-1][:k] 
            s_uncons = hat_v.argsort()[::-1][:k]
            s_uncons_alpha = hat_v_alpha.argsort()[::-1][:k]
            s_uncons_tau = s_opt 
            s_cons_er = get_constrained_solution(hat_v1, hat_v2, k//2, k//2, k)

            l1 = int(k * n1 / (n1 + n2))
            s_cons_pr = get_constrained_solution(hat_v1, hat_v2, l1, k - l1, k)

            s_cons_aq = get_constrained_solution(hat_v1, hat_v2, k - k_non_gen, k_non_gen, k)

            # Compute utilities of selection
            s_list = [s_cons_er, s_cons_aq, s_cons_pr, s_uncons_alpha, s_uncons_tau, s_opt, s_uncons]

            for i in range(len(u_list_)):
                assert(len(s_uncons) == len(s_list[i]))
                u_list_[i].append( np.sum(v[s_list[i]]) )

        for i in range(len(u_list)):
            u_list[i][key].append( np.mean([u_list_[i][j] / u_list_[-1][j] for j in range(ITER)]) )
            u_list_err[i][key].append( np.std([u_list_[i][j] / u_list_[-1][j] for j in range(ITER)]) / np.sqrt(ITER) )

        u_uncons[key].append( np.mean(u_list_[-1]) )
        u_uncons_err[key].append( np.std(u_list_[-1]) / np.sqrt(ITER) )
        
    for u, u_err in zip(u_list, u_list_err):
        u[key] = np.array(u[key])
        u_err[key] = np.array(u_err[key])

    plt.figure().clear()
        
    
    ########################################
    # # COMPARISON OF INTERVENTIONS
    ########################################
    plt.title(f'$n_1={n1}$, $n_2={n2}$, ITER={ITER}, '
                    + f'$\\alpha={np.round(alpha, 3)}$, $\\tau={np.round(tau, 3)}$', fontsize=20)
    
    linearize = lambda x: np.array(list(x.values())).reshape(-1)

    
    
    plt.errorbar(delta_alpha_list, linearize(u_cons_aq), yerr=linearize(u_cons_aq_err), linewidth = 6, label = f'Reservation' , linestyle='dashed')
    plt.errorbar(delta_alpha_list, linearize(u_uncons_alpha), yerr=linearize(u_uncons_alpha_err), linewidth = 6, label = f'Decrease skew ($|\\alpha-1|$)')
    

    if show_legend: plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=4, fontsize=30)
    
    
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.ylabel("Relative increase in utility", fontsize=34)
    plt.xlabel('$\\Delta_{\\alpha}$', fontsize=34)

    if save_fig:    
        pdf_savefig()
    else:   
        plt.show()
    plt.close()


    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################

    u_cons_er = {}
    u_cons_aq = {} # Actual quota
    u_cons_pr = {}
    u_opt = {}
    u_uncons_alpha = {}
    u_uncons_tau = {}
    u_uncons = {}
    u_list = [u_cons_er, u_cons_aq, u_cons_pr, u_uncons_alpha, u_uncons_tau, u_opt, u_uncons]

    u_cons_er_err = {}
    u_cons_aq_err = {}
    u_cons_pr_err = {}
    u_opt_err = {}
    u_uncons_alpha_err = {}
    u_uncons_tau_err = {}
    u_uncons_err = {}
    u_list_err = [u_cons_er_err, u_cons_aq_err, u_cons_pr_err, u_uncons_alpha_err, u_uncons_tau_err, u_opt_err, u_uncons_err]

    for delta_tau in tqdm(delta_tau_list):
        key = (delta_tau)
        for u, u_err in zip(u_list, u_list_err):
            u[key] = [] 
            u_err[key] = [] 

        if verbose: print('querying shifted tau...', flush=True)
        framework.set_tau(get_shifted_tau(tau, delta_tau))
        get_f = framework.get_biased_distribution()
        framework.set_tau(tau)

        pdf_biased_tau = np.array([get_f(v) for v in Omega])
        pdf_biased_tau = get_normalized_and_shifted(pdf_biased_tau, shift)
            
        u_list_ = [[] for i in range(len(u_list))] 

        for _ in range(ITER):    
            
            ########################################################################
            ## Sample utilities
            ########################################################################
            v1 = np.random.choice(size=n1, a=Omega, p=pdf_latent)
            hat_v1 = v1

            # Sample biased utilities and maintain a coupling between latent and (different) biased utilities 
            rand_bits = np.random.random(size = n1 + n2)
            v2 = Omega[ np.searchsorted( np.cumsum(pdf_latent), rand_bits) ]
            hat_v2 = Omega[ np.searchsorted( np.cumsum(pdf_biased), rand_bits) - 1 ]
            hat_v2_tau = Omega[ np.searchsorted( np.cumsum(pdf_biased_tau), rand_bits) - 1 ]

            # Merge utilities
            v = np.array( list(v1) + list(v2) )
            hat_v = np.array( list(hat_v1) + list(hat_v2) )
            hat_v_tau = np.array( list(hat_v1) + list(hat_v2_tau) )
            ########################################################################

            ########################################################################
            ## Compute outputs of different selection algorithms
            ########################################################################
            s_opt = v.argsort()[::-1][:k] 
            s_uncons = hat_v.argsort()[::-1][:k]
            s_uncons_alpha = s_opt 
            s_uncons_tau = hat_v_tau.argsort()[::-1][:k]
            s_cons_er = get_constrained_solution(hat_v1, hat_v2, k//2, k//2, k)

            l1 = int(k * n1 / (n1 + n2))
            s_cons_pr = get_constrained_solution(hat_v1, hat_v2, l1, k - l1, k)

            s_cons_aq = get_constrained_solution(hat_v1, hat_v2, k - k_non_gen, k_non_gen, k)

            # Compute utilities of selection
            s_list = [s_cons_er, s_cons_aq, s_cons_pr, s_uncons_alpha, s_uncons_tau, s_opt, s_uncons]

            for i in range(len(u_list_)):
                assert(len(s_uncons) == len(s_list[i]))
                u_list_[i].append( np.sum(v[s_list[i]]) )

        for i in range(len(u_list)):
            u_list[i][key].append( np.mean([u_list_[i][j] / u_list_[-1][j] for j in range(ITER)]) )
            u_list_err[i][key].append( np.std([u_list_[i][j] / u_list_[-1][j] for j in range(ITER)]) / np.sqrt(ITER) )

        u_uncons[key].append( np.mean(u_list_[-1]) )
        u_uncons_err[key].append( np.std(u_list_[-1]) / np.sqrt(ITER) )
        
    for u, u_err in zip(u_list, u_list_err):
        u[key] = np.array(u[key])
        u_err[key] = np.array(u_err[key])

    plt.figure().clear()
        
    
    ########################################
    # # COMPARISON OF INTERVENTIONS
    ########################################
    plt.title(f'$n_1={n1}$, $n_2={n2}$, ITER={ITER}, '
                    + f'$\\alpha={np.round(alpha, 3)}$, $\\tau={np.round(tau, 3)}$', fontsize=20)
    
    linearize = lambda x: np.array(list(x.values())).reshape(-1)

    
    plt.errorbar(delta_tau_list, linearize(u_cons_aq), yerr=linearize(u_cons_aq_err), linewidth = 6, label = f'Reservation' , linestyle='dashed')
    plt.errorbar(delta_tau_list, linearize(u_uncons_tau), yerr=linearize(u_uncons_tau_err), linewidth = 6, label = f'Decrease $\\tau$', color="green")

    if show_legend: plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=4, fontsize=30)
    
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.ylabel("Relative increase in utility", fontsize=34)
    plt.xlabel('$\\Delta_{\\tau}$', fontsize=34)

    if save_fig:    
        pdf_savefig()
    else:   
        plt.show()
    plt.close()

    return 


 
##############################################################################################################
#################################         Validation framework(s)         ####################################
##############################################################################################################

def generate_plots(latent_utilities = [0], biased_utilities =[0], pdf_latent = None, pdf_biased = None, Omega = None, err_func=err_func_l2, alpha_list=[2], tau_list=[-0.5], 
                   prior_list=[lebesgue_measure, get_pdf_normal(1, 1), get_pdf_normal(-0.25, 1), get_pdf_laplace(1, 1)], 
                   names_of_prior=['lebesgue', 'gaussian-mean=0|10', 'gaussian-mean=-0.25|10', 'pareto'], print_verbose=False, plot_prior=False, plot=True,
                   xlim = None, trunc_length = None, verbose = False, FAIL_CNT_THRESH=2, fname='', plot_fd=True):
    
    best_tv = 1e10 
    best_param = {'alpha': -1, 'tau': -1, 'f_0': -1, 'name': -1}

    print('initializing', flush=True)
    ############ Initialize underlying dist ##############
    if latent_utilities is not None:
        if trunc_length is None:
            Omega = get_omega([latent_utilities, biased_utilities])
            pdf_latent = get_pdf_from_utils(latent_utilities, omega=Omega)
            pdf_biased = get_pdf_from_utils(biased_utilities, omega=Omega)
        elif trunc_length < 0:
            Omega = get_omega([latent_utilities, biased_utilities])
            pdf_latent = get_pdf_from_utils(latent_utilities, omega=Omega)
            pdf_biased = get_pdf_from_utils(biased_utilities, omega=Omega)

            pdf_latent = pdf_latent[-trunc_length:] / sum(pdf_latent[-trunc_length:])
            pdf_biased = pdf_biased[-trunc_length:] / sum(pdf_biased[-trunc_length:])
            Omega = Omega[-trunc_length:]
        else:
            Omega = get_omega([latent_utilities, biased_utilities])
            pdf_latent = get_pdf_from_utils(latent_utilities, omega=Omega)[:trunc_length]
            pdf_biased = get_pdf_from_utils(biased_utilities, omega=Omega)[:trunc_length]
            Omega = Omega[:trunc_length]
    else:
        assert(pdf_latent is not None)    
        assert(pdf_biased is not None)    
        assert(Omega is not None)    

    ######################################################

    ################# Start making plot ##################
    plt.figure().clear()
    x = list(Omega) 
    if plot and plot_fd: plt.plot(x, pdf_latent, label='$f_D$ (real data)', linewidth=4)
    if plot: plt.plot(x, pdf_biased, label='Biased density (real-data)', linewidth=4)
    ######################################################

    fail_cnt = 0
    itert = tqdm(itertools.product(alpha_list, tau_list, zip(prior_list, names_of_prior)), 
                              total=len(alpha_list) * len(tau_list) * len(prior_list))
    

    for alpha, tau, (f_0, name) in itert: 
        if verbose: print(f'alpha={alpha}, tau={tau}, name={name}')
        itert.set_description(desc=f'alpha={np.round(float(alpha), 3)}, tau={np.round(float(tau), 2)}, name={name} || best_tv={np.round(float(best_tv), 3)}')

        try:        
            if verbose: print('normalizing...', flush=True)
            ################# Normalize and plot f0 ##################
            norm_f0_ = sum([f_0(v) * ONE for v in Omega]) * 1.0
            if f_0 == lebesgue_measure:
                norm_f0_ = 1
            f_0_normalized = lambda x: f_0(x) * 1.0 / norm_f0_

            if verbose: print('here1', flush=True)

            y = np.array([f_0_normalized(v) for i, v in enumerate(x)])
            if plot and plot_prior: plt.plot(x, y / sum(y), label=f'{name}', linewidth=4)
            ##########################################################

            if verbose: print('setting up integrator and framework...', flush=True)
            ################# Initialize framework ###################
            integrator = get_integrator(err_func, omega=Omega, pdf_v=pdf_latent)
            framework = Bias_Framework(alpha=alpha, tau=tau, get_lagrangian_integral=integrator, f_0=f_0_normalized, OMEGA=Omega)
            ##########################################################

            if verbose: print('querying...', flush=True)
            ############## Get biased distribution ################
            get_f = framework.get_biased_distribution(verbose=verbose)
            print('gamma: ', framework.gamma)
            ######################################################

            ################## Add to the plot ###################
            y = np.array([get_f(v) if not ISNAN(get_f(v)) else 0 for i, v in enumerate(x)])
            y /= sum(y)
            sumy = sum(y)

            z = np.zeros_like(y)
            best_tv_int = 1e10      
            best_shift = 1e10

            if verbose: print('computing shift...', flush=True)
            for s in range(-len(y)-1, len(y)+1, 1):
                get_shift(y, z, s)
                cur_tv_int = sum(abs(z / sumy - pdf_biased))
                
                if cur_tv_int < best_tv_int:
                    best_tv_int = cur_tv_int
                    best_shift = s 
            
            if verbose: print(f'best_shift={best_shift}, alpha={alpha}, tau={tau}')
            get_shift(y, z, best_shift)
            y = copy.deepcopy(z)


            if verbose: print('plot and compute TV...', flush=True)
            if plot: plt.plot(x, np.array(y) / sumy, label=f'$f^\\star$ with $\\tau={np.round(float(tau), 2)}$, $\\alpha={np.round(float(alpha), 2)}$', linewidth=4)
            if verbose: print('plot and compute TV2...', flush=True)

            if plot and print_verbose: plt.plot(x, abs(y / sumy - pdf_biased), label='abs-diff', linewidth=4)
            print('plot and compute TV3...', flush=True)
            print('*#'*50, flush=True)

            cur_tv = sum(abs(y / sumy - pdf_biased)) / 2
            print('computed TV: ', cur_tv)
            print(f'cur alpha={alpha} and tau={tau}')
            print('*#'*50)
            ###################################################### 

            if cur_tv < best_tv:
                best_tv = cur_tv
                best_param['alpha'] = alpha
                best_param['tau'] = tau
                best_param['f_0'] = f_0
                best_param['name'] = name
        
        except Exception as e: # work on python 3.x
            print('Error: '+ str(e))
            fail_cnt += 1
            print('failed')

        if fail_cnt >= FAIL_CNT_THRESH:
            break
        if plot: plt.legend(fontsize=14)


    if not plot and plot_fd: plt.plot(x, pdf_latent, label='$f_D$ (real data)', linewidth=4)
    if not plot: plt.plot(x, pdf_biased, label='Biased density (real-data)', linewidth=4)

    alpha = best_param['alpha']
    tau = best_param['tau']
    f_0 = best_param['f_0']
    name = best_param['name']

    if f_0 == -1:
        return

    if verbose: print('normalizing2...', flush=True)
    ################# Normalize and plot f0 ##################
    norm_f0_ = sum([f_0(v) * 1.0 for v in Omega]) * 1.0
    if f_0 == lebesgue_measure:
        norm_f0_ = 1
    f_0_normalized = lambda x: f_0(x) * 1.0 / norm_f0_

    y = np.array([f_0_normalized(v) for i, v in enumerate(x)])
    if not plot and plot_prior: plt.plot(x, y / sum(y), label=f'{name}', linewidth=4)
    ##########################################################

    if verbose: print('initializing2...', flush=True)
    ################# Initialize framework ###################
    integrator = get_integrator(err_func, omega=Omega, pdf_v=pdf_latent)
    framework = Bias_Framework(alpha=alpha, tau=tau,  
                                                            get_lagrangian_integral=integrator, f_0=f_0_normalized, OMEGA=Omega)
    ##########################################################

    if verbose: print('querying2...', flush=True)
    ############## Get biased distribution ################
    get_f = framework.get_biased_distribution()
    if verbose: print('gamma: ', framework.gamma)
    ######################################################

    if verbose: print('addingtoplot2...', flush=True)
    ################## Add to the plot ###################
    y = np.array([get_f(v) if not ISNAN(get_f(v)) else 0 for i, v in enumerate(x)])
    y /= sum(y)
    sumy = sum(y)

    z = np.zeros_like(y)
    best_tv_int = 1e10
    best_shift = 1e10

    for s in range(-len(y)-1, len(y)+1, 1):
        get_shift(y, z, s)
        cur_tv_int = sum(abs(z / sumy - pdf_biased))
        
        if cur_tv_int < best_tv_int:
            best_tv_int = cur_tv_int
            best_shift = s 

    get_shift(y, z, best_shift)
    y = copy.deepcopy(z)

    if not plot: plt.plot(x, np.array(y) / sumy, label=f'$f^\\star$ with $\\tau={np.round(float(tau), 2)}$, $\\alpha={np.round(float(alpha), 2)}$', linewidth=4, linestyle='dotted')

    if not plot and print_verbose: plt.plot(x, abs(y / sumy - pdf_biased), label='abs-diff', linewidth=4)
    print('*#'*50)
    cur_tv = 0
    for i in range(len(pdf_biased)):
        cur_tv += abs(y[i] / sumy - pdf_biased[i]) / 2
    
    print((abs(y[0] / sumy - pdf_biased[0] / sum(pdf_biased))) / 2, pdf_biased[0] / sum(pdf_biased), y[0] / sumy)
    print('computed best TV: ', cur_tv)
    print('computed KL-div: ', get_kl_div(y/sumy, pdf_biased))
    print('computed infty-div: ', get_l_inf_dist(y/sumy, pdf_biased))
    print(f'best_param: {best_param}')
    print(f'best_shift={best_shift}')
    print('*#'*50)
    ###################################################### 

    if xlim is not None:
        plt.xlim(xlim)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    pdf_savefig('best-fit'+fname)
    plt.show()

##############################################################################################################
##############################################################################################################



def generate_plots_of_means_and_variances(latent_utilities=[0], biased_utilities=[0], pdf_latent=None, pdf_biased=None, Omega=None, err_func=err_func_l2, alpha_list=[2], tau_list=[-0.5],
                            prior_list=[lebesgue_measure, get_pdf_normal(1, 1), get_pdf_normal(-0.25, 1), get_pdf_laplace(1, 1)], pdf_v=None,
                            names_of_prior=['lebesgue', 'gaussian-mean=0|10', 'gaussian-mean=-0.25|10', 'pareto'], print_verbose=False, plot_prior=False, plot=True,
                            xlim=None, trunc_length=None, verbose=False, FAIL_CNT_THRESH=2, integrator=None, is_discrete=True, framework=None, print_table=False):

    print('initializing', flush=True)
    ############ Initialize underlying dist ##############
    if latent_utilities is not None:
        if trunc_length is None:
            Omega = get_omega([latent_utilities, biased_utilities])
            pdf_latent = get_pdf_from_utils(latent_utilities, omega=Omega)
            pdf_biased = get_pdf_from_utils(biased_utilities, omega=Omega)
        elif trunc_length < 0:
            Omega = get_omega([latent_utilities, biased_utilities])
            pdf_latent = get_pdf_from_utils(latent_utilities, omega=Omega)
            pdf_biased = get_pdf_from_utils(biased_utilities, omega=Omega)

            pdf_latent = pdf_latent[-trunc_length:] / \
                sum(pdf_latent[-trunc_length:])
            pdf_biased = pdf_biased[-trunc_length:] / \
                sum(pdf_biased[-trunc_length:])
            Omega = Omega[-trunc_length:]
        else:
            Omega = get_omega([latent_utilities, biased_utilities])
            pdf_latent = get_pdf_from_utils(
                latent_utilities, omega=Omega)[:trunc_length]
            pdf_biased = get_pdf_from_utils(
                biased_utilities, omega=Omega)[:trunc_length]
            Omega = Omega[:trunc_length]
    elif not is_discrete:
        print()
    else:
        assert (pdf_latent is not None)
        assert (pdf_biased is not None)
        assert (Omega is not None)

    ######################################################

    ################# Start making plot ##################
    plt.figure().clear()
    if is_discrete:
        x = list(Omega)
        if plot:
            plt.plot(x, pdf_latent, label='$f_D$ (real data)', linewidth=4)
    else:
        x = np.linspace(Omega[0], Omega[1], 10000)
        x_tmp_mean = np.linspace(Omega[0], Omega[1]*10000, 10000)
        if plot:
            plt.plot(x, [pdf_v(z)
                     for z in x], label='$f_D$ (real data)', linewidth=4)
        print(
            f'true mean: {integrate(lambda x: pdf_v(x) * x, Omega[0], Omega[1]*1000, pieces=5)}')
        

    ######################################################

    means = {}
    variances = {}
    gammas = {}

    results = {}  # density values for alpha and x...

    fail_cnt = 0
    itert = tqdm(itertools.product(alpha_list, tau_list, zip(prior_list, names_of_prior)),
                 total=len(alpha_list) * len(tau_list) * len(prior_list))
    for alpha, tau, (f_0, name) in itert:
        if verbose:
            print(f'alpha={alpha}, tau={tau}, name={name}')
        itert.set_description(
            desc=f'alpha={np.round(float(alpha), 3)}, tau={np.round(float(tau), 2)}, name={name}')
        key = (np.round(alpha, 5), np.round(tau, 5))

        try:
            if verbose:
                print('normalizing...', flush=True)
            if is_discrete:
                ################# Normalize and plot f0 ##################
                norm_f0_ = sum([f_0(v) * ONE for v in Omega]) * 1.0
                if f_0 == lebesgue_measure:
                    norm_f0_ = 1

                def f_0_normalized(x): return f_0(x) * 1.0 / norm_f0_

                if verbose:
                    print('here1', flush=True)

                y = np.array([f_0_normalized(v) for i, v in enumerate(x)])
                if plot and plot_prior:
                    plt.plot(x, y / sum(y), label=f'{name}', linewidth=4)
                ##########################################################

                if verbose:
                    print('setting up integrator and framework...', flush=True)
                ################# Initialize framework ###################
                if integrator is None:
                    integrator = get_integrator(
                        err_func, omega=Omega, pdf_v=pdf_latent)
                framework = Bias_Framework(
                    alpha=alpha, tau=tau, get_lagrangian_integral=integrator, f_0=f_0_normalized, OMEGA=Omega)
                ##########################################################
            else:
                framework.set_alpha(alpha)
                framework.set_tau(tau)

            if verbose:
                print('querying...', flush=True)
            ############## Get biased distribution ################
            get_f = framework.get_biased_distribution(verbose=verbose)
            print('gamma: ', framework.gamma)
            ######################################################

            if is_discrete:
                ################## Add to the plot ###################
                y = np.array([get_f(v) for i, v in enumerate(x)])
                y *= 1/sum(y)

                results[alpha] = copy.deepcopy(y)
                means[key] = get_mean(y=y, Omega=Omega)

                variances[key] = get_variance(y=y, mean=means[key], Omega=Omega)
            else:
                y = np.array([get_f(v) for i, v in enumerate(x)])
                y *= 1/sum(y) 
                means[key] = integrate(lambda x: get_f(
                    x) * x, Omega[0], Omega[1]*1000, pieces=5) 
                print(f'mean is {means[key]} for alpha={alpha} and tau={tau}')

                variances[key] = integrate(lambda x: get_f(
                    x) * (x - means[key]) ** 2, Omega[0], Omega[1]*1000, pieces=5)

                x_tmp_tmp = list(range(1, 101))
                y_tmp = np.array([get_f(v) for i, v in enumerate(x_tmp_tmp)])
                y_tmp /= sum(y_tmp)
                results[alpha] = copy.deepcopy(y_tmp)

            gammas[key] = framework.gamma

            if is_discrete:
                if verbose:
                    print('plot and compute TV...', flush=True)
                if plot:
                    plt.plot(x, np.array(
                        y) / sum(y), label=f'$f^\\star$ with $\\tau={np.round(float(tau), 2)}$, $\\alpha={np.round(float(alpha), 2)}$', linewidth=4, alpha=0.5)
                if verbose:
                    print('plot and compute TV2...', flush=True)

                if plot and print_verbose:
                    plt.plot(x, abs(y / sum(y) - pdf_biased),
                             label='abs-diff', linewidth=4)
                if verbose:
                    print('plot and compute TV3...', flush=True)
                if verbose:
                    print('*#'*50, flush=True)
                cur_tv = sum(abs(y / sum(y) - pdf_biased)) / 2
                if verbose:
                    print('computed TV: ', cur_tv)
                if verbose:
                    print('*#'*50)
            else:
                if plot:
                    plt.plot(x, np.array(y) / sum(y) * len(x) / (
                        Omega[1] - Omega[0]), label=f'$f^\\star$ with $\\tau={np.round(float(tau), 2)}$, $\\alpha={np.round(float(alpha), 4)}$', linewidth=4, alpha=0.5)

        except Exception as e:  # work on python 3.x
            print('Error: ' + str(e))
            fail_cnt += 1
            print('failed')

        if fail_cnt >= FAIL_CNT_THRESH:
            break

        if plot:
            plt.legend(fontsize=14)

    if not is_discrete and print_table:
        x_tmp_tmp = list(range(1, 101)) 
        print('x,', end='')
        for alpha in alpha_list:
            print(alpha, end=",")

        print('\n')

        for i, x in enumerate(x_tmp_tmp):
            print(x, end=",")
            for alpha in alpha_list:
                print(results[alpha][i], end=",")

            print('\n')

    if xlim is not None:
        plt.xlim(xlim)
    plt.show()
    plt.close()

    plt.figure().clear()
    plt.title('Plot of $\\mathbb{E}_{x\\sim f}[x]$ vs. $\\alpha$', fontsize=22)

    plt.plot(alpha_list, [means[(np.round(alpha, 5), np.round(
        tau_list[0], 5))] for alpha in alpha_list], linewidth=4)

    plt.legend(fontsize=20)

    plt.ylabel('$\\mathbb{E}_{x\\sim f}[x]$', fontsize=22)
    plt.xlabel('$\\alpha$', fontsize=22)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    pdf_savefig()
    
    plt.show()
    plt.close()

    plt.figure().clear()
    plt.title('Plot of $\\mathbb{E}_{x\\sim f}[x]$ vs. $\\tau$', fontsize=22)
    
    plt.plot(tau_list, [means[(np.round(alpha_list[0], 5), np.round(
        tau, 5))] for tau in tau_list], linewidth=4)

    plt.legend(fontsize=20)

    plt.ylabel('$\\mathbb{E}_{x\\sim f}[x]$', fontsize=22)
    plt.xlabel('$\\tau$', fontsize=22)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    pdf_savefig()
    
    plt.show()
    plt.close()

    #############################################################################################

    plt.figure().clear()
    plt.title('Plot of Var${}_{x\\sim f}[x]$ vs. $\\alpha$', fontsize=22)
    
    plt.plot(alpha_list, [variances[(np.round(alpha, 5), np.round(
        tau_list[0], 5))] for alpha in alpha_list], linewidth=4)

    plt.legend(fontsize=20)

    plt.ylabel('Var${}_{x\\sim f}[x]$', fontsize=22)
    plt.xlabel('$\\alpha$', fontsize=22)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    pdf_savefig()
    
    plt.show()
    plt.close()

    plt.figure().clear()
    plt.title('Plot of Var${}_{x\\sim f}[x]$ vs. $\\tau$', fontsize=22)
    
    plt.plot(tau_list, [variances[(np.round(alpha_list[0], 5), np.round(
        tau, 5))] for tau in tau_list], linewidth=4)

    plt.legend(fontsize=20)

    plt.ylabel('Var${}_{x\\sim f}[x]$', fontsize=22)
    plt.xlabel('$\\tau$', fontsize=22)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    pdf_savefig()
    
    plt.show()
    plt.close()

    #############################################################################################

    plt.figure().clear()
    plt.title('Plot of $\\gamma^{\star}$ vs. $\\alpha$', fontsize=22)
    plt.plot(alpha_list, [gammas[(np.round(alpha, 5), np.round(
        tau_list[0], 5))] for alpha in alpha_list], linewidth=4)

    plt.legend(fontsize=20)

    plt.ylabel('$\\gamma^{\star}$', fontsize=22)
    plt.xlabel('$\\alpha$', fontsize=22)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    pdf_savefig()
    
    plt.show()
    plt.close()

##############################################################################################################

def generate_plots_of_means(latent_utilities = [0], biased_utilities =[0], pdf_latent = None, pdf_biased = None, Omega = None, err_func=err_func_l2, alpha_list=[2], tau_list=[-0.5], 
                   prior_list=[lebesgue_measure, get_pdf_normal(1, 1), get_pdf_normal(-0.25, 1), get_pdf_laplace(1, 1)], pdf_v = None,
                   names_of_prior=['lebesgue', 'gaussian-mean=0|10', 'gaussian-mean=-0.25|10', 'pareto'], print_verbose=False, plot_prior=False, plot=True,
                   xlim = None, trunc_length = None, verbose = False, FAIL_CNT_THRESH=2, integrator = None, is_discrete = True, framework=None, print_table=False):

    print('initializing', flush=True)
    ############ Initialize underlying dist ##############
    if latent_utilities is not None:
        if trunc_length is None:
            Omega = get_omega([latent_utilities, biased_utilities])
            pdf_latent = get_pdf_from_utils(latent_utilities, omega=Omega)
            pdf_biased = get_pdf_from_utils(biased_utilities, omega=Omega)
        elif trunc_length < 0:
            Omega = get_omega([latent_utilities, biased_utilities])
            pdf_latent = get_pdf_from_utils(latent_utilities, omega=Omega)
            pdf_biased = get_pdf_from_utils(biased_utilities, omega=Omega)

            pdf_latent = pdf_latent[-trunc_length:] / sum(pdf_latent[-trunc_length:])
            pdf_biased = pdf_biased[-trunc_length:] / sum(pdf_biased[-trunc_length:])
            Omega = Omega[-trunc_length:]
        else:
            Omega = get_omega([latent_utilities, biased_utilities])
            pdf_latent = get_pdf_from_utils(latent_utilities, omega=Omega)[:trunc_length]
            pdf_biased = get_pdf_from_utils(biased_utilities, omega=Omega)[:trunc_length]
            Omega = Omega[:trunc_length]
    elif not is_discrete:
        print()
    else:
        assert(pdf_latent is not None)    
        assert(pdf_biased is not None)    
        assert(Omega is not None)    

    ######################################################

    ################# Start making plot ##################
    plt.figure().clear()
    if is_discrete:
        x = list(Omega) 
        if plot: plt.plot(x, pdf_latent, label='$f_D$ (real data)', linewidth=4)
    else:
        x = np.linspace(Omega[0], Omega[1], 10000)
        x_tmp_mean = np.linspace(Omega[0], Omega[1]*10000, 10000)
        if plot: plt.plot(x, [pdf_v(z) for z in x], label='$f_D$ (real data)', linewidth=4)
        print(f'true mean: {integrate(lambda x: pdf_v(x) * x, Omega[0], Omega[1]*1000, pieces=5)}')
    

    ######################################################

    means = {}
    gammas = {}

    results = {} # density values for alpha and x... 

    fail_cnt = 0
    itert = tqdm(itertools.product(alpha_list, tau_list, zip(prior_list, names_of_prior)), 
                              total=len(alpha_list) * len(tau_list) * len(prior_list))
    for alpha, tau, (f_0, name) in itert: 
        if verbose: print(f'alpha={alpha}, tau={tau}, name={name}')
        itert.set_description(desc=f'alpha={np.round(float(alpha), 3)}, tau={np.round(float(tau), 2)}, name={name}')
        key = (np.round(alpha, 5), np.round(tau, 5))

        try:        
            if verbose: print('normalizing...', flush=True)
            if is_discrete:
                ################# Normalize and plot f0 ##################
                norm_f0_ = sum([f_0(v) * ONE for v in Omega]) * 1.0
                if f_0 == lebesgue_measure:
                    norm_f0_ = 1
                f_0_normalized = lambda x: f_0(x) * 1.0 / norm_f0_

                if verbose: print('here1', flush=True)

                y = np.array([f_0_normalized(v) for i, v in enumerate(x)])
                if plot and plot_prior: plt.plot(x, y / sum(y), label=f'{name}', linewidth=4)
                ##########################################################

                if verbose: print('setting up integrator and framework...', flush=True)
                ################# Initialize framework ###################
                if integrator is None:
                    integrator = get_integrator(err_func, omega=Omega, pdf_v=pdf_latent) 
                framework = Bias_Framework(alpha=alpha, tau=tau, get_lagrangian_integral=integrator, f_0=f_0_normalized, OMEGA=Omega)
                ##########################################################
            else:
                framework.set_alpha(alpha)
                framework.set_tau(tau)

            if verbose: print('querying...', flush=True)
            ############## Get biased distribution ################
            get_f = framework.get_biased_distribution(verbose=verbose)
            print('gamma: ', framework.gamma)
            ######################################################

            if is_discrete:
                ################## Add to the plot ###################
                y = np.array([get_f(v) for i, v in enumerate(x)])
                y *= 1/sum(y)

                results[alpha] = copy.deepcopy(y)
                means[key] = get_mean(y=y, Omega=Omega)
            else:
                y = np.array([get_f(v) for i, v in enumerate(x)])
                y *= 1/sum(y)
                
                means[key] = integrate(lambda x: get_f(x) * x, Omega[0], Omega[1]*1000, pieces=5)
                
                print(f'mean is {means[key]} for alpha={alpha} and tau={tau}')

                x_tmp_tmp = list(range(1, 101))
                y_tmp = np.array([get_f(v) for i, v in enumerate(x_tmp_tmp)])
                y_tmp /= sum(y_tmp)
                results[alpha] = copy.deepcopy(y_tmp)
            
            gammas[key] = framework.gamma

            if is_discrete:
                if verbose: print('plot and compute TV...', flush=True)
                if plot: plt.plot(x, np.array(y) / sum(y), label=f'$f^\\star$ with $\\tau={np.round(float(tau), 2)}$, $\\alpha={np.round(float(alpha), 2)}$', linewidth=4, alpha=0.5)
                if verbose: print('plot and compute TV2...', flush=True)

                if plot and print_verbose: plt.plot(x, abs(y / sum(y) - pdf_biased), label='abs-diff', linewidth=4)
                if verbose: print('plot and compute TV3...', flush=True)
                if verbose: print('*#'*50, flush=True)
                cur_tv = sum(abs(y / sum(y) - pdf_biased)) / 2
                if verbose: print('computed TV: ', cur_tv)
                if verbose: print('*#'*50)
            else: 
                if plot: plt.plot(x, np.array(y) / sum(y) * len(x) / (Omega[1] - Omega[0]), label=f'$f^\\star$ with $\\tau={np.round(float(tau), 2)}$, $\\alpha={np.round(float(alpha), 2)}$', linewidth=4, alpha=0.5)
        
        except Exception as e: # work on python 3.x
            print('Error: '+ str(e))
            fail_cnt += 1
            print('failed')

        if fail_cnt >= FAIL_CNT_THRESH:
            break
        
        if plot: plt.legend(fontsize=14)


    if not is_discrete and print_table:
        x_tmp_tmp = list(range(1, 101))
        
        print('x,', end='')
        for alpha in alpha_list:
            print(alpha, end=",")

        print('\n')

        for i, x in enumerate(x_tmp_tmp): 
            print(x, end=",")
            for alpha in alpha_list:
                print(results[alpha][i], end=",")

            print('\n')

    
    if xlim is not None:
        plt.xlim(xlim)
    plt.show()
    plt.close()
    
    plt.figure().clear()
    plt.title('Plot of $\\mathbb{E}_{x\\sim f}[x]$ vs. $\\alpha$', fontsize=22)
    
    plt.plot(alpha_list, [means[(np.round(alpha, 5), np.round(tau_list[0], 5))] for alpha in alpha_list], linewidth=4)
    
    
    
    plt.legend(fontsize=20)
    
    plt.ylabel('$\\mathbb{E}_{x\\sim f}[x]$', fontsize=22)
    plt.xlabel('$\\alpha$', fontsize=22)
    
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    pdf_savefig()
    
    plt.show()
    plt.close()



    plt.figure().clear()
    plt.title('Plot of $\\gamma^{\star}$ vs. $\\alpha$', fontsize=22)
    plt.plot(alpha_list, [gammas[(np.round(alpha, 5), np.round(tau_list[0], 5))] for alpha in alpha_list], linewidth=4)
     
    plt.legend(fontsize=20)
    
    plt.ylabel('$\\gamma^{\star}$', fontsize=22)
    plt.xlabel('$\\alpha$', fontsize=22)
    
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    pdf_savefig()
    
    plt.show()
    plt.close()





##############################################################################################################
####################################         Generate 3D plots...         ####################################
##############################################################################################################


def generate_plots_table_of_means_and_variance(latent_utilities = None, biased_utilities = None, pdf_latent = None, pdf_biased = None, 
                                               Omega = None, err_func=None, alpha_list=None, tau_list=None, pdf_v = None,
                                               integrator = None, is_discrete = True, framework=None, xlims = None,
                                               print_verbose=False, plot=True, verbose = False, FAIL_CNT_THRESH=None):

    f_0 = lebesgue_measure
     
    print('initializing', flush=True)
    ############ Initialize underlying dist ##############
    if not is_discrete:
        print()
    elif latent_utilities is not None:
        Omega = get_omega([latent_utilities, biased_utilities])
        pdf_latent = get_pdf_from_utils(latent_utilities, omega=Omega)
        pdf_biased = get_pdf_from_utils(biased_utilities, omega=Omega)
    else:
        assert(pdf_latent is not None)    
        assert(pdf_biased is not None)    
        assert(Omega is not None)    

    ######################################################

    ################# Start making plot ##################
    plt.figure().clear()
    if is_discrete:
        x = list(Omega) 
    else:
        x = np.linspace(Omega[0], 50, 10000)
        print(f'true mean: {integrate(lambda x: pdf_v(x) * x, Omega[0], Omega[1], pieces=5)}')
    
    ######################################################

    means = {}
    variances = {}
    gammas = {}

    fail_cnt = 0
    itert = tqdm(itertools.product(alpha_list, tau_list), 
                              total=len(alpha_list) * len(tau_list))
    for alpha, tau in itert: 
        if verbose: print(f'alpha={alpha}, tau={tau}')
        itert.set_description(desc=f'alpha={np.round(float(alpha), 3)}, tau={np.round(float(tau), 2)}')

        key = (np.round(alpha, 5), np.round(tau, 5))

        
        if verbose: print('normalizing...', flush=True)
        if is_discrete:
            ################# Initialize framework ###################
            if verbose: print('setting up integrator and framework...', flush=True)
            if integrator is None:
                integrator = get_integrator(err_func, omega=Omega, pdf_v=pdf_latent) 
            framework = Bias_Framework(alpha=alpha, tau=tau, get_lagrangian_integral=integrator, f_0=f_0, OMEGA=Omega)
            ##########################################################
        else:
            framework.set_alpha(alpha)
            framework.set_tau(tau)

        if verbose: print('querying...', flush=True)
        
        ############## Get biased distribution ################
        get_f = framework.get_biased_distribution(verbose=verbose)
        print('gamma: ', framework.gamma)
        ######################################################


        if is_discrete:
            ################## Add to the plot ###################
            y = np.array([get_f(v) for i, v in enumerate(x)])
            y *= 1/sum(y)

            means[key] = get_mean(y=y, Omega=Omega)
            variances[key] = get_variance(y=y, mean=means[key], Omega=Omega)
        else:
            y = np.array([get_f(v) for i, v in enumerate(x)])
            y *= 1/sum(y)

            means[key] = integrate(lambda x: get_f(x) * x, Omega[0], Omega[1], pieces=5)
            variances[key] = integrate(lambda x: get_f(x) * (x - means[key])**2, Omega[0], Omega[1], pieces=5)
            
            print(f'mean is {means[key]} for alpha={alpha} and tau={tau}')
            print(f'variance is {variances[key]} for alpha={alpha} and tau={tau}')

        gammas[key] = framework.gamma

        if is_discrete:
            if verbose: print('plot and compute TV...', flush=True)
            if plot: plt.plot(x, np.array(y) / sum(y), label=f'$f^\\star$ with $\\tau={np.round(float(tau), 2)}$, $\\alpha={np.round(float(alpha), 2)}$', linewidth=4, alpha=0.5)
            if verbose: print('plot and compute TV2...', flush=True)

            if plot and print_verbose: plt.plot(x, abs(y / sum(y) - pdf_biased), label='abs-diff', linewidth=4)
            if verbose: print('plot and compute TV3...', flush=True)
            if verbose: print('*#'*50, flush=True)
            cur_tv = sum(abs(y / sum(y) - pdf_biased)) / 2
            if verbose: print('computed TV: ', cur_tv)
            if verbose: print('*#'*50)
        else: 
            if plot: plt.plot(x, np.array(y) / sum(y) * len(x) / (Omega[1] - Omega[0]), label=f'$f^\\star$ with $\\tau={np.round(float(tau), 2)}$, $\\alpha={np.round(float(alpha), 2)}$', linewidth=4, alpha=0.5)

        if fail_cnt >= FAIL_CNT_THRESH:
            break
        
        if plot: 
            plt.legend(fontsize=14)
            if xlims is not None: plt.xlim(xlims)


    def print_table(dict_, alpha_list, tau_list):
        print(',', end='')
        for alpha in alpha_list:
            print(f'alpha={alpha}', end=",")

        print('\n', end='')

        for i, tau in enumerate(tau_list): 
            print(f'tau={-tau}', end=",")
            for alpha in alpha_list:
                key = (np.round(alpha, 5), np.round(tau, 5))
                print(dict_[key], end=",")
            print('\n', end='')

    print("%"*50)
    print("MEANS:")
    print_table(means, alpha_list, tau_list)
    print("%"*50)

    print("%"*50)
    print("VARIANCES:")
    print_table(variances, alpha_list, tau_list)
    print("%"*50)

    print("%"*50)
    print("GAMMA:")
    print_table(gammas, alpha_list, tau_list)
    print("%"*50)
 
