import random 
import pickle, os
import numpy as np
import mpmath as mp
 
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import rcParams
import datetime
import string

from tqdm.notebook import tqdm
from collections import Counter

USE_LONG_MATH = False
ONE = 1.0 if USE_LONG_MATH else 1.0
# np.mpf(1.0)
E = mp.e if USE_LONG_MATH else np.e
SQRT = lambda x: mp.sqrt(x) if USE_LONG_MATH else np.sqrt(float(x))
EXP = lambda x: mp.exp(x) if USE_LONG_MATH else np.exp(float(x))
ISNAN = lambda x: mp.isnan(x) if USE_LONG_MATH else np.isnan(float(x))
LOG = lambda x: mp.log(x) / mp.log(mp.e)if USE_LONG_MATH else (np.log(float(x)) / np.log(np.e) if x > 0 else 0)


home_folder = './figures'

############################################################################################
#################################    Matplotlib helpers     ################################
############################################################################################

rcParams.update({
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
        'figure.figsize': (10,6),
})

def file_str():
    """ Auto-generates file name."""
    now = datetime.datetime.now()
    return now.strftime("%y_%m_%d-")

rand_string = lambda length: ''.join(random.choice(string.ascii_lowercase) for i in range(length))

def pdf_savefig(fname=''):
    """ Saves figures as pdf """
    fname = file_str() + fname + '-' + rand_string(5)
    plt.savefig(home_folder+f"/{fname}.pdf", bbox_inches="tight")

def eps_savefig():
    """ Saves figure as encapsulated postscript file (vector format)
        so that it isn't pixelated when we put it into a pdf. """
    pdf_savefig()



 
############################################################################################
##############################     Miscellaneous helpers     ###############################
############################################################################################

def write(obj, name):
    if os.path.isfile(f'{name}.obj'):
        print("File with the same name already exists!")
        raise Exception 
    file = open(f"{name}.obj","wb")
    pickle.dump(obj, file)
    file.close() 

def read_obj(name):
    if not os.path.isfile(f'{name}.obj'):
        print("File with this name does not exist!")
        raise Exception 
    file = open(f"{name}.obj","rb")
    obj = pickle.load(file)
    file.close()
    return obj

def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

def append_hatv_to_xticks(ax):
    locs, labels = plt.xticks()
    for l in labels:
        if l.get_text()[0] == '-':
            l.set_text('$\\widehat{v}$ - ' + l.get_text()[1:])
        elif l.get_text() == "0":
            l.set_text('$\\widehat{v}$')
        else:
            l.set_text('$\\widehat{v}$ + ' + l.get_text())

    ax.set_xticks(locs, labels)

def get_shift(y, z, s):
    for i in range(len(y)):
        if i+s >= 0 and i+s < len(y):
            z[i] = y[i + s]
        else:
            z[i] = 0

############################################################################################
###################################     Math helpers     ###################################
############################################################################################


def integrate(func, MIN = -10, MAX = 10, is_discrete = False, Omega = [], verbose=False, pieces=5):
    '''
        Integrate func from MIN to MAX 
        Breaks integral into 50 peices, which heuristically speeds up computation 
    '''
    if is_discrete:
        tot = 0 * ONE
        tot = sum([(func(v) if not ISNAN(func(v)) else 0)  * (abs(func(v)) > 1e-300) * (abs(func(v)) < 1e500) for v in Omega])
        
        if verbose: print(f'\t\t\t Fin integral...')
        
        return tot
    
    
    ITER = pieces
    x = np.linspace(MIN, MAX, ITER)
    intervals = [ [x[i], x[i+1]] for i in range(ITER - 1)]

    tot = 0
    for i in intervals:
        tmp = quad(func, i[0], i[1])  
        if not ISNAN(tmp[0]):
            tot += tmp[0] 

    return tot 

def get_l_inf_dist(pdf_1, pdf_2):
    if type(pdf_1) != type(np.zeros(2)): pdf_1 = np.array(pdf_1)
    if type(pdf_2) != type(np.zeros(2)): pdf_2 = np.array(pdf_2)
    maxa = 0
    for p1, p2 in zip(pdf_1, pdf_2):
        maxa = max(maxa, LOG(p1 / p2) if p1 > 0 and p2 > 0 else 0)
    return maxa 

def get_kl_div(pdf_1, pdf_2):
    if type(pdf_1) != type(np.zeros(2)): pdf_1 = np.array(pdf_1)
    if type(pdf_2) != type(np.zeros(2)): pdf_2 = np.array(pdf_2)
    tota = 0
    for p1, p2 in zip(pdf_1, pdf_2):
        tota += p1 * LOG(p1 / p2) if p1 > 0 and p2 > 0 else 0
    return tota

def get_tv(pdf_1, pdf_2):
    if type(pdf_1) != type(np.zeros(2)): pdf_1 = np.array(pdf_1)
    if type(pdf_2) != type(np.zeros(2)): pdf_2 = np.array(pdf_2)
    return sum(abs(pdf_1 - pdf_2)) / 2


##############################     Pdf functions helpers     ###############################

lebesgue_measure = lambda x: ONE

def get_pdf_pareto(param):
    def pdf_pareto(x, param):
        if x >= 1: 
            return param / x ** (param + 1)
        else:
            return 0
    return lambda x: pdf_pareto(x, param=param)

def get_pdf_normal(mean, std):
    assert(std > 0)
    return lambda x: EXP(-(x - mean)**2 / 2 / std / std) / SQRT(2*np.pi) / std

def get_pdf_laplace(mean, b):
    assert(b > 0)
    return lambda x: EXP(-abs(x - mean) / b) / 2 / b

def get_pdf_expon(lamb):
    def pdf_laplace(x, lamb):
      if x > 0:
        return lamb * EXP(-lamb * x)
      else:
        return 0
    
    return lambda x: pdf_laplace(x, lamb)

def get_pdf_uniform(a, b):
   assert( b > a)
   return lambda x: (x >= a) * (x <= b) * 1/(b-a)
    

def get_pdf_uniform_omega(omega):
    return lambda x: 1 / len(omega)

##############################     Error function helpers     ##############################

def err_func_l2(v, hat_v, alpha):    
    if hat_v <= v:
        tmp = ONE * (v-hat_v) ** 2
    else: 
        tmp = alpha * ONE *  (v-hat_v) ** 2 
    return tmp
    
def err_func_l1(v, hat_v, alpha):
    if hat_v <= v:
        return abs(v-hat_v) 
    else: 
        return alpha * abs(v-hat_v) 
    
def err_func_exp(v, hat_v, alpha):
    if hat_v <= v:
        return E ** (v-hat_v)
    else: 
        return alpha * E ** (v-hat_v)

def err_func_pareto(v, hat_v, alpha):
    if hat_v <= v: 
        return LOG(hat_v/v) if v > 0 else 0
    else:
        return alpha * LOG(hat_v / v) if v > 0 else 0
    
def err_func_expon(v, hat_v, alpha):
    if hat_v <= v:
        return hat_v/v
    else: 
        return alpha * hat_v / v  
    
def get_err_func_custom(pdf_latent, f_0, Omega):
    map_ = { v:i for i, v in enumerate(Omega)}
    def err_func_custom(v, hat_v, alpha):       
        tmp = pdf_latent[map_[hat_v]] / pdf_latent[map_[hat_v]] / f_0(hat_v)      

        if hat_v <= v:  
            return LOG ( tmp )
        else: 
            return LOG( alpha * tmp )
    
    return lambda v, hat_v, alpha: err_func_custom(v=v, hat_v=hat_v, alpha=alpha)

############################     Framework specific helpers     ############################

def get_omega(list_of_lists):
    Omega_set = set()
    for util in list_of_lists:
        Omega_set = Omega_set.union(    np.unique(util)     )
    
    return np.array(sorted(Omega_set))

def get_pdf_from_utils(utils, omega):
    cntr = Counter(utils)
    pdf_v = np.array([cntr[v] * ONE for v in omega])
    return pdf_v / sum(pdf_v) * ONE

def get_integrator(loss, omega, pdf_v, trunc_length = None) -> lambda float: float:
    '''
        Returns a function which computes integral 
        $$\int_\Omega \ell_v(hat_v) d\mu_D(v)$$
        where \mu_D is the empirical density defined by latent_utilities
        TODO
    '''

    assert(len(pdf_v) == len(omega))
    
    if trunc_length == None:
        return lambda hat_v, alpha: sum([loss(omega[i], hat_v, alpha) * p for i, p in enumerate(pdf_v)])
    else:
        return lambda hat_v, alpha: sum([loss(omega[i], hat_v, alpha) * p for i, p in enumerate(pdf_v[:trunc_length])])

def get_integrator_continuous(loss, pdf_v, omega) -> lambda float: float:
    def integrator(hat_v, alpha):
        I = lambda v: loss(v=v, hat_v=hat_v, alpha=alpha) * pdf_v(v)
        return integrate(I, omega[0], omega[1])
    return integrator