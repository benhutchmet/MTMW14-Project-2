import numpy as np
import matplotlib.pyplot as plt
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
    
    # compute the denominator
    denominator = math.exp(b) - math.exp(a)
    
    return numerator/denominator
    
def analytic_solution(params_analytic):
    """Analytic solver for the SWEs using equations (3), (4) and (5) from project brief specifying the solutions at (x, y) using methods from Mushgrave (1985).
    
    Inputs:
    params - the dictionary containing the constants to be used (in this case 'params_analytic')
    
    Outputs:
    x - array of x values
    y - array of y values
    u - analytic solution for fluid motion in x-direction
    v - analytic solution for fluid motion in y-direction
    eta - analytic solution for the deviation of water surface from its initial level.
    """

    # extract the number of x and y points from the dictionary
    x_points = params_analytic['x_points']
    y_points = params_analytic['y_points']

    # establish the gridbox size
    gridbox_size = params_analytic['gridbox_size']

    # define arrays for x and y
    x = np.linspace(0, x_points*gridbox_size, x_points)
    y = np.linspace(0, y_points*gridbox_size, y_points)

    # extract the constants from the dictionary
    f0 = params_analytic['f0']
    beta = params_analytic['beta']
    g = params_analytic['g']
    gamma = params_analytic['gamma']
    rho = params_analytic['rho']
    H = params_analytic['H']
    tau0 = params_analytic['tau0']
    L = params_analytic['L']
    eta0 = params_analytic['eta0']

    # define the arrays to store u, v and eta results
    # does the value of v being situated on the x-axis //
    # and the value of u being situated on the y-axis affect this?
    u = np.zeros((x_points, y_points))
    v = np.zeros((x_points, y_points))
    eta = np.zeros((x_points, y_points))

    # start the analysis by computing epsilon
    epsilon = epsilon_function_analytic(gamma, L, beta)

    # then compute the a and b coefficients
    # for use in functions f_1 and f_2
    a = a_function_analytic(epsilon)
    b = b_function_analytic(epsilon)

    # define the coefficient containing tau_0
    # to make code in loops easier to read
    tau_coeff = tau0 / (np.pi*gamma*rho*H)

    # compute u, v and eta for all values of x and y
    for i in range(x_points): # i for x
        for j in range(y_points): # j for y

            # improve readability
            sin = np.sin
            cos = np.cos
            pi = np.pi

            # compute the analytic solutions of //
            # u, v and eta at [x,y]
            # what is the deal with j and i indexing here?

            # analytic solution for u[x,y]
            u[i, j] = -tau_coeff * f1_function_analytic(x[i]/L, a, b) * cos(pi*y[j]/L)

            # analytic solution for v[x,y]
            v[i, j] = tau_coeff * f2_function_analytic(x[i]/L, a, b) * sin(pi*y[j]/L)

            # analytic solution for eta[x,y]
            eta[i, j] = eta0 + tau_coeff * (f0*L/g) * (
                gamma/(f0*pi) * f2_function_analytic(x[i]/L, a, b) * cos(pi*y[j]/L)
                + 1/pi * f1_function_analytic(x[i]/L, a, b) * (
                sin(pi*y[j]/L) * (1 + beta*y[j]/f0)
                + beta*L/(f0*pi) * cos(pi*y[j]/L)
                    )
                )

    # return the values for the analytic solution of u, v and eta as well as x and y
    return u, v, eta, x, y




