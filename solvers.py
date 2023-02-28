# %%

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
    numerator = (math.exp(a) - 1)*b*math.exp(b*x) + (1 - math.exp(b))*a*math.exp(a*x)
    
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
    x = np.arange(x_points + 1)*gridbox_size
    y = np.arange(y_points + 1)*gridbox_size

    # extract the constants from the dictionary
    f0 = params_analytic['f0']
    beta = params_analytic['beta']
    g = params_analytic['g']
    gamma = params_analytic['gamma']
    rho = params_analytic['rho']
    H = params_analytic['H']
    tau0 = params_analytic['tau0']
    L = params_analytic['L']
    #eta0 = params_analytic['eta0']

    eta0 = 0.007437890509284137 # value of eta at eta[0, L/2] as an estimate of eta0

    # define the arrays to store u, v and eta results
    u = np.zeros((y_points, x_points))
    v = np.zeros((y_points, x_points))
    eta = np.zeros((y_points, x_points))

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
    for j in range(y_points): # j for y
        for i in range(x_points): # i for x

            # improve readability
            sin = np.sin
            cos = np.cos
            pi = np.pi

            # compute the analytic solutions of //
            # u, v and eta at [x,y]
            # what is the deal with j and i indexing here?

            # analytic solution for u[x,y]
            u[j, i] = -tau_coeff * f1_function_analytic(x[i]/L, a, b) * cos(pi*y[j]/L)

            # analytic solution for v[x,y]
            v[j, i] = tau_coeff * f2_function_analytic(x[i]/L, a, b) * sin(pi*y[j]/L)

            # analytic solution for eta[x,y]
            eta[j, i] = eta0 + tau_coeff * (f0*L/g) * (
                gamma/(f0*pi) * f2_function_analytic(x[i]/L, a, b) * cos(pi*y[j]/L)
                + 1/pi * f1_function_analytic(x[i]/L, a, b) * (
                sin(pi*y[j]/L) * (1 + beta*y[j]/f0)
                + (beta*L)/(f0*pi) * cos(pi*y[j]/L)
                    )
                )

    # print the value of eta at the centre of the domain
    #print('eta at centre of domain: ', eta[0, int(x_points/2)])

    # return the values for the analytic solution of u, v and eta as well as x and y
    return u, v, eta, x, y

# define a function for the plotting in Task C
def plotting_taskC(params_analytic):
    """Function for plotting the results of the analytic solution for the ocean gyre simulation.
    
    Inputs:
    params_analytic - the dictionary containing the constants to be used (in this case 'params_analytic')
    
    Outputs:
    None
    """
    
    # compute the analytic solution
    u, v, eta, x, y = analytic_solution(params_analytic)
    
    # plot the results as three seperate plots, with a colourbar for each

    # plot the u results
    plt.figure()
    plt.pcolormesh(x/1000, y/1000, u, cmap='RdBu_r')
    plt.colorbar()
    plt.title('Analytic solution for u')
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    # save the figure with the title specified in dictionary as as a .png file
    plt.savefig(params_analytic['u_fig_name'] + '.png')
    plt.show()

    # plot the v results
    plt.figure()
    plt.pcolormesh(x/1000, y/1000, v, cmap='RdBu_r')
    plt.colorbar()
    plt.title('Analytic solution for v')
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    # save the figure with the title specified in dictionary as as a .png file
    plt.savefig(params_analytic['v_fig_name'] + '.png')
    plt.show()

    # plot the eta results
    plt.figure()
    plt.pcolormesh(x/1000, y/1000, eta, cmap='RdBu_r')
    plt.colorbar()
    plt.title('Analytic solution for eta')
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    # save the figure with the title specified in dictionary as as a .png file
    plt.savefig(params_analytic['eta_fig_name'] + '.png')
    plt.show()


# now test the function
plotting_taskC(params_analytic_highest_res)

# print the values of the constants used in the simulation
#print(params_analytic_highest_res)

# %%
    
# now we move onto task D where we consider a forward-backward time scheme (Matsuno (1966); Beckers and Deleersnijder (1993))
#  this method alternates the order in which the two momentum equations are solved
# first - u before v
# then - v before u
# then returned back to u before v and so on

# before we can do this we must first specify the domain and the grid on which this simulation occurs

# but before we can do that we must first define the zonal wind forcing for the simulation

def zonal_wind_stress(y_ugrid, x_points, L, tau0):
    """Function for computing the zonal wind stress at the u-grid points using distance from origin.
    
    Inputs:
    y_ugrid - the y-coordinates of the u-grid points
    x_points - the number of x-grid points
    L - the length of the domain
    tau0 - the maximum wind stress
    
    Outputs:
    tau - the zonal wind stress at the u-grid points
    """

    # compute zonal wind stress at the u-grid points mapped onto y
    tau = tau0*-np.cos(np.pi*y_ugrid/L)

    # return the zonal wind stress mapped onto the arakawa u-grid (c) using the np.tile function
    # DOES THIS NEED TO BE TRANSPOSED?
    return np.tile(tau, (x_points + 1, 1))

# we also need to compute coriolis and map this onto both the u-grid and the v-grid
def coriolis(y_ugrid, y_vgrid, x_points, f0, beta):
    """Function for computing the coriolis parameter at the u-grid and v-grid points using distance from origin.
    
    Inputs:
    y_ugrid - the y-coordinates of the u-grid points
    y_vgrid - the y-coordinates of the v-grid points
    x_points - the number of x-grid points
    f0 - the reference coriolis parameter
    beta - the gradient of the coriolis parameter
    
    Outputs:
    f_u - the coriolis parameter at the u-grid points
    f_v - the coriolis parameter at the v-grid points
    """

    # calcuate the coriolis parameter at the u-grid points mapped onto y
    f_u = f0 + (beta*y_ugrid)
    
    # calculate the coriolis parameter at the v-grid points mapped onto y
    f_v = f0 + (beta*y_vgrid)

    # map the coriolis parameter onto the arakawa u-grid (c) using the np.tile function
    f_u = np.tile(f_u, (x_points + 1, 1))

    # map the coriolis parameter onto the arakawa v-grid (c) using the np.tile function
    f_v = np.tile(f_v, (x_points, 1))

    # return the coriolis parameter at the u-grid and v-grid points
    return f_u, f_v

# we need to set up the grid such that the values for meridional velocity (v) are computed onto a corresponding index in the zonal velocity (u) on an arawaka c-grid

# we also need to set up the grid such that the values for zonal velocity (u) are computed onto a corresponding index in the meridional velocity (v) on an arawaka c-grid

# set up the function v to u grid mapping

def v_to_ugrid_mapping(v, y_points):
    """Function for mapping the v-grid values onto the u-grid.
    
    Inputs:
    v - the v-grid values
    y_points - the number of y-grid points
    
    Outputs:
    index of v mapped onto the u-grid set up for forward-backward time scheme
    """

    # create array of zeros for the v-grid values mapped onto the u-grid
    # we use this to append zeros to the start/end of v arrays
    zeros_array_v_to_u = np.zeros((y_points, 1))

    # set up the slicing for forward-backward time scheme
    v_identical_index = np.concatenate((v[:-1, :], zeros_array_v_to_u), axis=1)
    v_j_minus_1_index = np.concatenate((zeros_array_v_to_u, v[:-1, :]), axis=1)
    v_jadd1_i = np.concatenate((v[1:, :], zeros_array_v_to_u), axis=1)
    v_jadd1_iminus1 = np.concatenate((zeros_array_v_to_u, v[1:, :]), axis=1)

    # return the index of v mapped onto the u-grid set up for forward-backward time scheme
    return (v_identical_index + v_j_minus_1_index + v_jadd1_i + v_jadd1_iminus1)/4

# set up the function u to v grid mapping
def u_to_vgrid_mapping(u, x_points):
    """Function for mapping the u-grid values onto the v-grid.
    
    Inputs:
    u - the u-grid values
    x_points - the number of x-grid points
    
    Outputs:
    index of u mapped onto the v-grid set up for forward-backward time scheme
    """

    # create array of zeros for the u-grid values mapped onto the v-grid
    # we use this to append zeros to the start/end of u arrays
    zeros_array_u_to_v = np.zeros((1, x_points))

    # set up the slicing for forward-backward time scheme
    u_identical_index = np.concatenate((u[:, :-1], zeros_array_u_to_v), axis=0)
    u_j_iplus1 = np.concatenate((u[:, 1:], zeros_array_u_to_v), axis=0)
    u_j_minus1_i = np.concatenate((zeros_array_u_to_v, u[:, :-1]), axis=0)
    u_j_minus1_iplus1 = np.concatenate((zeros_array_u_to_v, u[:, 1:]), axis=0)

    # return the index of u mapped onto the v-grid set up for forward-backward time scheme
    return (u_identical_index + u_j_iplus1 + u_j_minus1_i + u_j_minus1_iplus1)/4

# define a function to calculate the gradient of eta on the arakawa c-grid for both the u grid and the v grid

def eta_gradient(eta, x_points, y_points, dx, dy):
    """Function for computing the gradient of eta on the arakawa c-grid for both the u grid and the v grid.
    
    Inputs:
    eta - the free surface height
    x_points - the number of x-grid points
    y_points - the number of y-grid points
    dx - the grid spacing in the x-direction
    dy - the grid spacing in the y-direction
    
    Outputs:
    deta_dx - the gradient of eta in the x-direction
    deta_dy - the gradient of eta in the y-direction
    """
    # create arrays of zeros for the gradient of eta in the x-direction and y-direction
    array_zeros_x = np.zeros((y_points, 1))
    array_zeros_y = np.zeros((1, x_points))

    # concatenate the zeros to the start/end of the eta array for the x-direction
    eta_x = np.concatenate((eta, array_zeros_x), axis=1)
    eta_j_iminus1 = np.concatenate((array_zeros_x, eta), axis=1)

    # concatenate the zeros to the start/end of the eta array for the y-direction
    eta_y = np.concatenate((eta, array_zeros_y), axis=0)
    eta_jminus1_i = np.concatenate((array_zeros_y, eta), axis=0)

    # calculate the gradient of eta in the x-direction
    deta_dx = (eta_x - eta_j_iminus1)/(dx)
    
    # calculate the gradient of eta in the y-direction
    deta_dy = (eta_y - eta_jminus1_i)/(dy)

    # return the gradient of eta in the x-direction and y-direction
    return deta_dx, deta_dy

# we want to compute the energy to test the stability of the model

def energy(u, v, eta, dx, dy, rho, H, g):
    """Function for computing the energy of the model.
    
    Inputs:
    u - the zonal velocity
    v - the meridional velocity
    eta - the free surface height
    dx - the grid spacing in the x-direction
    dy - the grid spacing in the y-direction
    rho - the density of the fluid
    H - the depth of the fluid
    g - the gravitational acceleration
    
    Outputs:
    energy - the total energy of the model over the domain.
    """
    # calculate the kinetic energy
    kinetic_energy = H*(u**2 + v**2)
    
    # calculate the potential energy
    potential_energy = g*eta**2
    
    # calculate the total energy per grid point
    energy = 1/2*rho*(kinetic_energy + potential_energy)
    
    # return the total energy over the whole domain
    return np.sum(energy)*dx*dy

# define the function to solve the shallow water equations using the forward-backward time scheme
def forward_backward_time_scheme(params):
    """Function for solving the shallow water equations using the forward-backward time scheme.
    
    Inputs:
    params - a dictionary containing the parameters for the model
    
    Outputs:
    u - the zonal velocity
    v - the meridional velocity
    eta - the free surface height
    """
    # set up the parameters






# %%
