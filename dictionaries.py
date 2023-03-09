"""Dictionaries containing the constants and paramaters needed for the tasks"""


# For the analytic solution
params_analytic = {'f0': 10**-4, # coriolis paramater (s^-1)
		           'beta': 10**-11, # beta value for B-plane approx (m^-1s^-1)	
                   'g': 10, # gravitational acceleration (ms^-2)
                   'gamma': 10**-6, # linear drag coefficient (s^-1)
                   'rho': 1000, # uniform density (kg m^-3)
                   'H': 1000, # resting depth of fluid (m)
                   'tau0': 0.2, # wind stress vector constant (N m^-2)
                   'L': 10**6, # dimensions of square domain (m)
                   'eta0': -0.0068, # unknown constant of integration -
                   # UPDATE //
                   # WITH STEADY STATE ONCE ANALYTICAL SOLUTION HAS BEEN RUN
                   'gridbox_size': 100000, # spacing of square grid cells (m)
                   'x_points': 10, # number of points in the x domain
                   'y_points': 10, # number of points in the y domain
                   'u_fig_name': 'u_analytic', # name for the u figure
                   'v_fig_name': 'v_analytic', # name for the v figure
                   'eta_fig_name': 'eta_analytic'} # name for the eta figure
                   
params_analytic_higher_res = {'f0': 10**-4, # coriolis paramater (s^-1)
                                'beta': 10**-11, # beta value for B-plane approx (m^-1s^-1)
                                'g': 10, # gravitational acceleration (ms^-2)
                                'gamma': 10**-6, # linear drag coefficient (s^-1)
                                'rho': 1000, # uniform density (kg m^-3)
                                'H': 1000, # resting depth of fluid (m)
                                'tau0': 0.2, # wind stress vector constant (N m^-2)
                                'L': 10**6, # dimensions of square domain (m)
                                'eta0': -0.0068, # unknown constant of integration -
                                # updated with value at steady state eta(0,L/2)
                                'gridbox_size': 50000, # spacing of square grid cells (m)
                                'x_points': 20, # number of points in the x domain
                                'y_points': 20, # number of points in the y domain
                                'u_fig_name': 'u_analytic_higher_res', # name for the u figure
                                'v_fig_name': 'v_analytic_higher_res', # name for the v figure
                                'eta_fig_name': 'eta_analytic_higher_res'} # name for the eta figure

params_analytic_highest_res = {'f0': 10**-4, # coriolis paramater (s^-1)
                                'beta': 10**-11, # beta value for B-plane approx (m^-1s^-1)
                                'g': 10, # gravitational acceleration (ms^-2)
                                'gamma': 10**-6, # linear drag coefficient (s^-1)
                                'rho': 1000, # uniform density (kg m^-3)
                                'H': 1000, # resting depth of fluid (m)
                                'tau0': 0.2, # wind stress vector constant (N m^-2)
                                'L': 10**6, # dimensions of square domain (m)
                                'eta0': -0.0068, # unknown constant of integration -
                                # updated with value at steady state eta(0,L/2)
                                'gridbox_size': 10000, # spacing of square grid cells (m)
                                'x_points': 100, # number of points in the x domain
                                'y_points': 100, # number of points in the y domain
                                'u_fig_name': 'u_analytic_highest_res', # name for the u figure
                                'v_fig_name': 'v_analytic_highest_res', # name for the v figure
                                'eta_fig_name': 'eta_analytic_highest_res'} # name for the eta figure


# For the numerical solution

params_numerical_TaskD_1Day = {'f0': 10**-4, # coriolis paramater (s^-1)
                                 'beta': 10**-11, # beta value for B-plane approx (m^-1s^-1)
                                    'g': 10, # gravitational acceleration (ms^-2)
                                    'gamma': 10**-6, # linear drag coefficient (s^-1)
                                    'rho': 1000, # uniform density (kg m^-3)
                                    'H': 1000, # resting depth of fluid (m)
                                    'tau0': 0.2, # wind stress vector constant (N m^-2)
                                    'tau_meridional': 0, # meridional wind stress (N m^-2)
                                    'L': 10**6, # dimensions of square domain (m)
                                    'dx': 100000, # x grid spacing (m)
                                    'dy': 100000, # y grid spacing (m)
                                    'x_points': 10, # number of points in the x domain
                                    'y_points': 10, # number of points in the y domain
                                    'dt': 480, # time step (s) - ambitious (approx 0.7 below CFL criteria)
                                    'nt': int(86400/480), # number of time steps - for 1 day
                                    'u_fig_name': 'u_numerical_TaskD_1Day', # name for the u figure
                                    'v_fig_name': 'v_numerical_TaskD_1Day', # name for the v figure
                                    'eta_fig_name': 'eta_numerical_TaskD_1Day', # name for the eta figure
                                    'eta_contour_fig_name': 'eta_contour_numerical_TaskD_1Day', # name for the 2D eta contour figure
                                    'use_higher_resolution': 'False', # use higher resolution grid'
                                    'use_highest_resolution': 'False', # use highest resolution grid'
                                    'task': 'D1' # Task D1
                                    }

params_numerical_TaskD_SteadyState = {'f0': 10**-4, # coriolis paramater (s^-1)
                                    'beta': 10**-11, # beta value for B-plane approx (m^-1s^-1)
                                    'g': 10, # gravitational acceleration (ms^-2)
                                    'gamma': 10**-6, # linear drag coefficient (s^-1)
                                    'rho': 1000, # uniform density (kg m^-3)
                                    'H': 1000, # resting depth of fluid (m)
                                    'tau0': 0.2, # wind stress vector constant (N m^-2)
                                    'tau_meridional': 0, # meridional wind stress (N m^-2)
                                    'L': 10**6, # dimensions of square domain (m)
                                    'dx': 100000, # x grid spacing (m)
                                    'dy': 100000, # y grid spacing (m)
                                    'x_points': 10, # number of points in the x domain
                                    'y_points': 10, # number of points in the y domain
                                    'dt': 480, # time step (s) - ambitious (approx 0.7 below CFL criteria)
                                    'nt': int(100 * 86400/480), # number of time steps - for 50 days
                                    'u_fig_name': 'u_numerical_TaskD_SteadyState', # name for the u figure
                                    'v_fig_name': 'v_numerical_TaskD_SteadyState', # name for the v figure
                                    'eta_fig_name': 'eta_numerical_TaskD_SteadyState', # name for the eta figure
                                    'eta_contour_fig_name': 'eta_contour_numerical_TaskD_SteadyState', # name for the 2D eta contour figure
                                    'use_higher_resolution': 'False', # use higher resolution grid'
                                    'use_highest_resolution': 'False', # use highest resolution grid
                                    'task': 'D2' # Task D2
                                    }

# create a dictionary of parameters for the numerical solution for half the resolution
params_numerical_TaskD_SteadyState_highres_50 = params_numerical_TaskD_SteadyState | {'dx': 50000, # x grid spacing (m) 
                                                                                      'dy': 50000, # y grid spacing (m) 
                                                                                      'x_points': 20, # number of points in the x domain 
                                                                                      'y_points': 20, # number of points in the y domain 
                                                                                      'use_higher_resolution': 'True', # use higher resolution grid' , 
                                                                                      'use_highest_resolution': 'False', # use highest resolution grid'
                                                                                      'task': 'E2' # Task D2 
                                                                                      'dt': 240, # time step (s) - ambitious (approx 0.7 below CFL criteria)
                                                                                      'nt': int(100 * 86400/240), # number of time steps - for 50 days}
                                                                                      'u_fig_name': 'u_numerical_highres_50', # name for the u figure
                                                                                      'v_fig_name': 'v_numerical_highres_50', # name for the v figure
                                                                                      'eta_fig_name': 'eta_numerical_highres_50', # name for the eta figure
                                                                                      'eta_contour_fig_name': 'eta_contour_numerical_highres_50', # name for the 2D eta contour figure
}

# create a dictionary of parameters for the numerical solution for the highest resolution - 10km
params_numerical_TaskD_SteadyState_highres_10 = params_numerical_TaskD_SteadyState | {'dx': 10000, # x grid spacing (m)
                                                                                       'dy': 10000, # y grid spacing (m)
                                                                                       'x_points': 100, # number of points in the x domain
                                                                                       'y_points': 100, # number of points in the y domain
                                                                                       'use_higher_resolution': 'False', # use higher resolution grid
                                                                                       'use_highest_resolution': 'True', # use highest resolution grid
                                                                                       'task': 'E3' # Task D2
                                                                                       'dt': 40, # time step (s) - conservative (approx 0.6 below CFL criteria)
                                                                                       'nt': int(100 * 86400/40), # number of time steps - for 50 days
                                                                                       'u_fig_name': 'u_numerical_highres_10', # name for the u figure
                                                                                       'v_fig_name': 'v_numerical_highres_10', # name for the v figure
                                                                                       'eta_fig_name': 'eta_numerical_highres_10', # name for the eta figure
                                                                                       'eta_contour_fig_name': 'eta_contour_numerical_highres_10', # name for the 2D eta contour figure
}

params_numerical_TaskD_differences = params_numerical_TaskD_SteadyState | {'u_fig_name': 'u_numerical_TaskD_differences', # name for the u figure
                                    'v_fig_name': 'v_numerical_TaskD_differences', # name for the v figure
                                    'eta_fig_name': 'eta_numerical_TaskD_differences', # name for the eta figure
                                    'task': 'D3' # Task D3
                                    }

params_numerical_TaskE_energy = params_numerical_TaskD_SteadyState | {
                                    'task': 'E', # Task E energy
                                    'energy_fig_name': 'energy_TaskE', # name for the energy figure
                                    'energy_difference_fig_name': 'energy_difference_TaskE', # name for the energy differences figure
                                    }

params_numerical_TaskE_energy_test = {'tau0': 0.2,  # Wind stress constant
                           'tau_meridional': 0,  # meridional wind stress (assumed zero for now)
                           'f0': 10 ** -4,  # Coriolis parameter (s-1)
                           'beta': 10 ** -11,  # Beta value (m-1s-1)
                           'g': 10,  # Gravitational constant (ms-2)
                           'gamma': 10 ** -6,  # Linear drag coefficient (s-1)
                           'rho': 1000,  # Seawater density (kgm-3)
                           'H': 1000,  # Resting depth of fluid, assumed constant (m)
                           'L': 10 ** 6,  # Length of domain (m)
                           'dt': 50,  # time step (conservative based pon average flow speeds in WBC- Stommel)
                           'nt': int(86400/50)*50,  # Number of timesteps (to ensure a 1 day model runtime)
                           'x_points': 100,  # number of grid points in x
                           'y_points': 100,  # number of grid points in y
                           'dx': 10 ** 4,  # grid spacing in x
                           'dy': 10 ** 4,  # grid spacing in y
                           'eta_fig_name': 'eta_t=50days',  # name of saved figure for eta output
                           'u_fig_name': 'u_t=50days',  # name of saved figure for u output
                           'v_fig_name': 'v_t=50days',
                           'use_higher_resolution': 'True',
                           'task': 'E',
                           'eta_contour_fig_name': 'eta_contour_plot'}  # name of saved figure for v output

  

