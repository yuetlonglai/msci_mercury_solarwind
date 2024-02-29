import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mercury_simulation import Mercury_B_induced_sim

def prelim_data(df,select=False):
    # some more cleaning
    df = df.drop('Unnamed: 0', axis=1)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['Pram'] = 1.67e-27 * df['np1'] * 100**3 * (df['vp1'] * 1e3)**2
    df['absB'] = (df['Bx']**2 + df['By']**2 + df['Bz']**2)**0.5
    df = df[(df['Pram'] != 0)]
    df['date'] = df['datetime'].dt.date
    df.sort_values('datetime',inplace=True)
    k_value = 50
    df['Rstdoff'] = 1+k_value * df['Pram']**(-1/6) / 2439.7 # in unit of radius of Mercury R_M
    return df
def time_select(df,b,e):
    # select time interval
    # begin = '1977-04-01 00:00:00'
    # end = '1977-04-10 23:59:59'
    begin=b
    end=e
    df_select = df[(df['datetime'] >= begin) & (df['datetime'] <= end)]
    df_select.sort_values('datetime',inplace=True)
    return df_select

# world = Mercury_B_induced_sim()
# world.bfield_inducing([30*1e-9],cartesian=False).bfield_plot(cartesian_label=False)

num_points = 10
x=np.linspace(0,2*np.pi,num_points)
P = 30e-9*(np.sin(x))+40e-9
plt.ylabel(r'$P_{ram}$ (nPa)')
plt.plot(x,P,'-',color='black',label='Toy Data')
plt.legend()
plt.show()
world2 = Mercury_B_induced_sim()
world2_inducing = world2.delta_bfield_inducing(P)#.bfield_plot(label='spherical_delta')
# g[n,m],h[n,m]

initial_guess_g = np.zeros((4,4))
initial_guess_h = np.zeros((4,4))
initial_guess_gauss = [initial_guess_g,initial_guess_h]
# world2_inducing.bfield_rms_residual_plot()
world2_minimised = world2_inducing.bfield_coefficient_minimisation(initial_guess_gauss)#.bfield_plot(label='spherical_delta')
# print(world2_minimised.minimisation_result)
world2_induced = world2_minimised.bfield_induced()#.bfield_plot(label='spherical')
# world2_induced.bfield_plot_3D()


# trajectory_r = 2*np.cosh(np.linspace(-np.pi/2,np.pi/2,num_points))
trajectory_r = 2*np.ones(num_points)
trajectory_theta = np.ones(num_points)*np.pi/2
trajectory_phi = np.linspace(-np.pi/2,np.pi/2,num_points)
# trajectory_phi = np.linspace(0,2*np.pi,num_points)
# trajectory_r = 4*(1-0.5**2)/(1+0.5*np.cos(trajectory_phi))
trajectory = np.column_stack([trajectory_r,trajectory_theta,trajectory_phi])
world2_induced.spacecraft_simulation(trajectory,plotting=True)