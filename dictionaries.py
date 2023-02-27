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
                   'eta0': 0.02495, # unknown constant of integration -
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
                                'eta0': 0.02495, # unknown constant of integration -
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
                                'eta0': 0.02495, # unknown constant of integration -
                                # updated with value at steady state eta(0,L/2)
                                'gridbox_size': 5000, # spacing of square grid cells (m)
                                'x_points': 200, # number of points in the x domain
                                'y_points': 200, # number of points in the y domain
                                'u_fig_name': 'u_analytic_highest_res', # name for the u figure
                                'v_fig_name': 'v_analytic_highest_res', # name for the v figure
                                'eta_fig_name': 'eta_analytic_highest_res'} # name for the eta figure
