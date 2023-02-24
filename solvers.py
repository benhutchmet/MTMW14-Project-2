import numpy as np
import matplotlib.plt as plt
import math
from dictionaries import *

# Define the analytical solvers for Task C
# Define the simple functions first

def a_function_analytic(epsilon):
    """Computes the a coefficient for calculation of the analytic solution of the ocean gyre simulation from Mushgrave (1985).
    
    Inputs:
    epsilon - constant
    
    Outputs:
    value of a coefficient.
    """
    
    return (-1 - np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)
    
def b_function_analytic(epsilon):
    """Computes the a coefficient for calculation of the analytic solution of the ocean gyre simulation from Mushgrave (1985).    
    
    Inputs:
    epsilon - constant
    
    Outputs:
    value of b coefficient.
    """
    
    return (-1 + np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)
    
def epsilon_function_analytic(gamma, L, beta):
    """Computes the value of episilon for the calculation of the analytic solution for the ocean gyre simulation from Mushgrave (1985).
    
    Inputs:
    gamma - linear drag coefficient (s^-1)
    L - dimensions of computational domain (m)
    beta - constant used for B-plane approximation (m^-1 s^-1)
    
    Outputs:
    value of episilon.
    """
    
    return gamma / (L*beta)
    
# now define the functions f_1 and f_2 which use the simple functions above

def f1_function_analytic(x, a, b):
    """f_1 function for calculating the analytical solution of the ocean gyre using methods from Mushgrave (1985).

    Inputs:
    x - the value of the x domain
    a - the a coefficient
    b - the b coefficient
    
    Outputs:
    value of f1 (at value of x).
    """
    
    # compute the numerator
    numerator = (math.exp(a) - 1)*b*math.exp(b*x) + (1 - math.exp(b))*math.exp(a*x)
    
    # compute the denominator
    denominator = math.exp(b) - math.exp(a)
    
    return np.pi*(1 + numerator/denominator)
    
def f2_function_analytic(x, a, b):
    """f_2 function for calculating the analytical solution of the ocean gyre using methods from Mushgrave (1985).

    Inputs:
    x - the value of the x domain
    a - the a coefficient
    b - the b coefficient
    
    Outputs:
    value of f2 (at value of x).
    """
    
    # compute the numerator
    numerator = (math.exp(a) - 1)*b*math.exp(b*x) + (1 - math.exp(b))*math.exp(a*x)