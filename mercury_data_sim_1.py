import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KT17_python_smz16 import KT17_python
from scipy import interpolate
from matplotlib import animation
from scipy import optimize
from scipy import special

def generate_sphere_points(radius, num_points):
    """
    Generate points on the surface of a sphere using the Fibonacci lattice method.

    :param radius: Radius of the sphere.
    :param num_points: Number of points to generate.
    :param spherical: If True, return points in spherical coordinates (latitude, longitude).
    :return: Array of points in Cartesian or spherical coordinates.
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        points.append((radius * x, radius * y, radius * z))

    return np.array(points)
def cartesian_to_spherical(x, y, z):
    """ coordinate transformation from cartesian to spherical coordinates """
    radius = np.sqrt(x**2 + y**2 + z**2)
    azimuthal_angle = np.arccos(z / radius)
    polar_angle = np.arctan2(y, x)
    return radius, azimuthal_angle, polar_angle
def spherical_to_latlon(radius, azimuthal_angle, polar_angle):
    """ express points on surface in lattitude and longtitude from spherical coordinates """
    latitude = np.pi / 2 - azimuthal_angle
    longitude = polar_angle
    return longitude, latitude
def bfield_cartesian_to_spherical(bx,by,bz):
    # Calculate radial distance
    bx = np.array(bx)
    by = np.array(by)
    bz = np.array(bz)
    br = np.sqrt(bx**2 + by**2 + bz**2)
    # Calculate polar angle (theta)
    btheta = np.arccos(bz / br)
    # Calculate azimuthal angle (phi)
    bphi = np.arctan(by, bx)
    return br, btheta, bphi
def bfield_spherical_to_cartesian(br,btheta,bphi,t,p):
    bx = br*np.sin(t)*np.cos(p) + btheta*np.cos(t)*np.cos(p) - bphi*np.sin(p)
    by = br*np.sin(t)*np.sin(p) + btheta*np.cos(t)*np.sin(p) + bphi*np.cos(p)
    bz = br*np.cos(t) - btheta*np.sin(t)
    return bx,by,bz

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

def bfield_interpolation(xval,yval,Bval,num_interp):
    # for plotting
    grid_x, grid_y = np.meshgrid(np.linspace(min(xval), max(xval), num_interp), np.linspace(min(yval), max(yval), num_interp))
    grid_z = interpolate.griddata((xval,yval),Bval,(grid_x,grid_y),method='cubic')
    return grid_x,grid_y,grid_z

def mapping_spherical_surface(n=1000):
    # map the desired spherical surface
    N = n
    R_M = 2439.7
    cords = generate_sphere_points(1740/R_M, N)
    X = cords[:,0]
    Y = cords[:,1]
    Z = cords[:,2] - 479/R_M # offsetting the magnetic dipole centre from planet's centre since they're different
    # changing it to lattitude and longtitude to plot in hte mollweide projection 
    R, theta, phi = cartesian_to_spherical(X, Y, Z)
    long, lat = spherical_to_latlon(R, theta, phi)
    return lat, long, X, Y, Z, R, theta, phi

def generate_bfield_external(ram_pres_series,cartesian=False):
    # map the desired spherical surface
    N=1000
    lat,long,X,Y,Z,R,phi,theta = mapping_spherical_surface(N)
    # start looping through a timeseries data of Pram
    ram_pres_series = list(np.array(ram_pres_series)*1e9)
    for i in range(len(ram_pres_series)):
        pram=ram_pres_series[i] # in nPa
        # Calculating the B-field using the kt17_bfield function
        model = KT17_python(Pram=pram)
        Bx, By, Bz = model.kt17_bfield(N, X, Y, Z)
        # Calculate the B-field magnitude for each point
        if cartesian == True:
            b_magnitude = np.sqrt(np.array(Bx)**2 + np.array(By)**2 + np.array(Bz)**2)
        else:
            Bx, By, Bz = bfield_cartesian_to_spherical(Bx,By,Bz) #x -> r, y -> theta, z -> phi
            b_magnitude = np.sqrt(np.array(Bx)**2 + np.array(By)**2 + np.array(Bz)**2)
    return Bx,By,Bz,b_magnitude,lat,long

def bfield_timeseries(ram_pres_series,cartesian=True):
    N = 1000
    # map the desired spherical surface
    lat,long,X,Y,Z,R,phi,theta = mapping_spherical_surface(N)
    # start looping through a timeseries data of Pram
    ram_pres_series = list(np.array(ram_pres_series)*1e9)
    for i in range(len(ram_pres_series)):
        pram=ram_pres_series[i] # in nPa
        # Calculating the B-field using the kt17_bfield function
        model = KT17_python(Pram=pram)
        Bx, By, Bz = model.kt17_bfield(N, X, Y, Z)
        # Calculate the B-field magnitude for each point
        if cartesian == True:
            b_magnitude = np.sqrt(np.array(Bx)**2 + np.array(By)**2 + np.array(Bz)**2)
        else:
            Bx, By, Bz = bfield_cartesian_to_spherical(Bx,By,Bz)
            b_magnitude = np.sqrt(np.array(Bx)**2 + np.array(By)**2 + np.array(Bz)**2)

        interp_num = 110 # number of points = square this number
        Bx_interp = bfield_interpolation(long,lat,Bx,interp_num)
        By_interp = bfield_interpolation(long,lat,By,interp_num)
        Bz_interp = bfield_interpolation(long,lat,Bz,interp_num)
        B_interp = bfield_interpolation(long,lat,b_magnitude,interp_num)

        subplot_positions = [[0,0],[0,1],[1,0],[1,1]]
        # plotting 3d plot of all points and the field vector
        fig,ax = plt.subplots(2,2,figsize=(10,7),subplot_kw={"projection": "mollweide"})
        fig.suptitle(r'$P_{ram} = $' + f'{pram:.3f} (nPa), ' + r'$R_{SS} = $' + f'{model.rss:.3f} '+r'($R_{M}$)')
        # coordinates = ax[0,0].scatter(long,lat,c=b_magnitude,cmap='seismic')
        # coordinates_x=ax[0,1].scatter(long,lat,c=Bx,cmap='seismic')
        # coordinates_y=ax[1,0].scatter(long,lat,c=By,cmap='seismic')
        # coordinates_z=ax[1,1].scatter(long,lat,c=Bz,cmap='seismic')
        coordinates = ax[0,0].contourf(B_interp[0],B_interp[1],B_interp[2],cmap='seismic',levels=80,extend='both')
        coordinates_x = ax[0,1].contourf(Bx_interp[0],Bx_interp[1],Bx_interp[2],cmap='seismic',levels=80,extend='both')
        coordinates_y = ax[1,0].contourf(By_interp[0],By_interp[1],By_interp[2],cmap='seismic',levels=80,extend='both')
        coordinates_z = ax[1,1].contourf(Bz_interp[0],Bz_interp[1],Bz_interp[2],cmap='seismic',levels=80,extend='both')
        if cartesian == True:
            fig.colorbar(coordinates,label=r'$|B|$ (nT)',orientation='horizontal',ax=ax[0,0])
            fig.colorbar(coordinates_x,label=r'$B_x$ (nT)',orientation='horizontal',ax=ax[0,1])
            fig.colorbar(coordinates_y,label=r'$B_y$ (nT)',orientation='horizontal',ax=ax[1,0],format=lambda x, _: f"{x:.2f}")
            fig.colorbar(coordinates_z,label=r'$B_z$ (nT)',orientation='horizontal',ax=ax[1,1])
        else:
            fig.colorbar(coordinates,label=r'$|B|$ (nT)',orientation='horizontal',ax=ax[0,0])
            fig.colorbar(coordinates_x,label=r'$B_r$ (nT)',orientation='horizontal',ax=ax[0,1])
            fig.colorbar(coordinates_y,label=r'$B_{\theta}$ (nT)',orientation='horizontal',ax=ax[1,0],format=lambda x, _: f"{x:.2f}")
            fig.colorbar(coordinates_z,label=r'$B_{\phi}$ (nT)',orientation='horizontal',ax=ax[1,1])
        for i in subplot_positions:
            ax[i[0],i[1]].grid(alpha=0.5)
        fig.tight_layout()
        plt.show()
    return None

def bfield_timeseries_animation(ram_pres_series):
    # map the desired spherical surface
    N = 1000
    cords = generate_sphere_points(1740/2439.7, N)
    X = cords[:,0]
    Y = cords[:,1]
    Z = cords[:,2] - 479/2439.7
    R, phi, theta = cartesian_to_spherical(X, Y, Z)
    lat, long = spherical_to_latlon(R, phi, theta)
    # Calculating the B-field using the kt17_bfield function
    ram_pres_series = list(np.array(ram_pres_series)*1e9)
    fig, ax = plt.subplots(2,2,figsize=(10,7),subplot_kw={"projection": "mollweide"})
    suptitle = fig.suptitle('')
    artist = []
    for i in range(len(ram_pres_series)):
        pram=ram_pres_series[i] # in nPa
        model = KT17_python(Pram=pram)
        Bx, By, Bz = model.kt17_bfield(N, X, Y, Z)
        # Calculate the B-field magnitude for each point
        b_magnitude = np.sqrt(np.array(Bx)**2 + np.array(By)**2 + np.array(Bz)**2)

        interp_num = 110 #number of points = square this number
        Bx_interp = bfield_interpolation(long,lat,Bx,interp_num)
        By_interp = bfield_interpolation(long,lat,By,interp_num)
        Bz_interp = bfield_interpolation(long,lat,Bz,interp_num)
        B_interp = bfield_interpolation(long,lat,b_magnitude,interp_num)

        subplot_positions = [[0,0],[0,1],[1,0],[1,1]]
        for i in subplot_positions:
            ax[i[0],i[1]].grid(alpha=0.5)
        # plotting 3d plot of all points and the field vector
        suptitle.set_text(r'$P_{ram} = $' + f'{pram:.3f} (nPa), ' + r'$R_{SS} = $' + f'{model.rss:.3f} '+r'($R_{M}$)')
        coordinates = ax[0,0]
        coordinates_x = ax[0,1]
        coordinates_y = ax[1,0]
        coordinates_z = ax[1,1]
        # coordinates.clear()
        # coordinates_x.clear()
        # coordinates_y.clear()
        # coordinates_z.clear()
        # coordinates = ax[0,0].scatter(long,lat,c=b_magnitude,cmap='seismic')
        # coordinates_x=ax[0,1].scatter(long,lat,c=Bx,cmap='seismic')
        # coordinates_y=ax[1,0].scatter(long,lat,c=By,cmap='seismic')
        # coordinates_z=ax[1,1].scatter(long,lat,c=Bz,cmap='seismic')
        coordinates = ax[0,0].contourf(B_interp[0],B_interp[1],B_interp[2],cmap='seismic',levels=80,extend='both')
        coordinates_x = ax[0,1].contourf(Bx_interp[0],Bx_interp[1],Bx_interp[2],cmap='seismic',levels=80,extend='both')
        coordinates_y = ax[1,0].contourf(By_interp[0],By_interp[1],By_interp[2],cmap='seismic',levels=80,extend='both')
        coordinates_z = ax[1,1].contourf(Bz_interp[0],Bz_interp[1],Bz_interp[2],cmap='seismic',levels=80,extend='both')
        # cbar1 = fig.colorbar(coordinates,label=r'$|B|$ (nT)',orientation='horizontal',ax=ax[0,0])
        # cbar2 = fig.colorbar(coordinates_x,label=r'$B_x$ (nT)',orientation='horizontal',ax=ax[0,1])
        # cbar3 = fig.colorbar(coordinates_y,label=r'$B_y$ (nT)',orientation='horizontal',ax=ax[1,0],format=lambda x, _: f"{x:.2f}")
        # cbar4 = fig.colorbar(coordinates_z,label=r'$B_z$ (nT)',orientation='horizontal',ax=ax[1,1])

        fig.tight_layout()
        artist.append([coordinates.collections,coordinates_x.collections,coordinates_y.collections,coordinates_z.collections])#,cbar1,cbar2,cbar3,cbar4])
    ani = animation.ArtistAnimation(fig=fig,artists=artist,repeat=True,interval=100,blit=True) 
    # ani.save('/Users/gordonlai/Documents/ICL/ICL_Y4/MSci_Mercury/msci_mercury_solarwind/bfield_test.gif',writer=animation.FFMpegWriter())
    plt.show()
    return ani

def bfield_plot(long,lat,Bx,By,Bz,cartesian=False):
    interp_num = 110 # number of points = square this number
    Bx_interp = bfield_interpolation(long,lat,Bx,interp_num)
    By_interp = bfield_interpolation(long,lat,By,interp_num)
    Bz_interp = bfield_interpolation(long,lat,Bz,interp_num)
    B_interp = bfield_interpolation(long,lat,np.sqrt(np.array(Bx)**2 + np.array(By)**2 + np.array(Bz)**2),interp_num)

    subplot_positions = [[0,0],[0,1],[1,0],[1,1]]
    # plotting 3d plot of all points and the field vector
    fig,ax = plt.subplots(2,2,figsize=(10,7),subplot_kw={"projection": "mollweide"})
    # fig.suptitle(r'$P_{ram} = $' + f'{pram:.3f} (nPa), ' + r'$R_{SS} = $' + f'{model.rss:.3f} '+r'($R_{M}$)')
    # coordinates = ax[0,0].scatter(long,lat,c=b_magnitude,cmap='seismic')
    # coordinates_x=ax[0,1].scatter(long,lat,c=Bx,cmap='seismic')
    # coordinates_y=ax[1,0].scatter(long,lat,c=By,cmap='seismic')
    # coordinates_z=ax[1,1].scatter(long,lat,c=Bz,cmap='seismic')
    coordinates = ax[0,0].contourf(B_interp[0],B_interp[1],B_interp[2],cmap='seismic',levels=80,extend='both')
    coordinates_x = ax[0,1].contourf(Bx_interp[0],Bx_interp[1],Bx_interp[2],cmap='seismic',levels=80,extend='both')
    coordinates_y = ax[1,0].contourf(By_interp[0],By_interp[1],By_interp[2],cmap='seismic',levels=80,extend='both')
    coordinates_z = ax[1,1].contourf(Bz_interp[0],Bz_interp[1],Bz_interp[2],cmap='seismic',levels=80,extend='both')
    if cartesian == True:
        fig.colorbar(coordinates,label=r'$|B|$ (nT)',orientation='horizontal',ax=ax[0,0])
        fig.colorbar(coordinates_x,label=r'$B_x$ (nT)',orientation='horizontal',ax=ax[0,1])
        fig.colorbar(coordinates_y,label=r'$B_y$ (nT)',orientation='horizontal',ax=ax[1,0],format=lambda x, _: f"{x:.2f}")
        fig.colorbar(coordinates_z,label=r'$B_z$ (nT)',orientation='horizontal',ax=ax[1,1])
    else:
        fig.colorbar(coordinates,label=r'$|B|$ (nT)',orientation='horizontal',ax=ax[0,0])
        fig.colorbar(coordinates_x,label=r'$B_r$ (nT)',orientation='horizontal',ax=ax[0,1])
        fig.colorbar(coordinates_y,label=r'$B_{\theta}$ (nT)',orientation='horizontal',ax=ax[1,0],format=lambda x, _: f"{x:.2f}")
        fig.colorbar(coordinates_z,label=r'$B_{\phi}$ (nT)',orientation='horizontal',ax=ax[1,1])
    for i in subplot_positions:
        ax[i[0],i[1]].grid(alpha=0.5)
    fig.tight_layout()
    plt.show()
    return None

def bfield_rms(coeffs, r, t, p, br_actual, bt_actual, bp_actual):
    g01,g11,g02,g12,g22,h11,h12,h22  = coeffs[0],coeffs[1],coeffs[2],coeffs[3],coeffs[4],coeffs[5],coeffs[6],coeffs[7]
    br_model, bt_model, bp_model = magnetic_field_analytic(r,t,p,A,g01,g11,g02,g12,g22,h11,h12,h22,True)
    return np.sum(np.sqrt((br_model-br_actual)**2 + (bt_model-bt_actual)**2 + (bp_model-bp_actual)**2))

def bfield_coefficient_minimisation(init_guess,r,t,p,bra,bta,bpa):
    minimise = optimize.minimize(bfield_rms,x0=init_guess,args=(r,t,p,bra,bta,bpa))
    return minimise.x


A=1
# bfield_timeseries_animation(np.array([100,80,60,40,20,10])*1e-9)
# bfield_timeseries([30*1e-9],cartesian=False)
lat,long,X,Y,Z,r,theta,phi = mapping_spherical_surface(1000)
# B_actual = generate_bfield_external([30*1e-9])
# coeffs = bfield_coefficient_minimisation([-35, 1, -3, 0, 0, 13, 0, 0, 0],A,phi,theta,B_actual[0],B_actual[1],B_actual[2])
# results = magnetic_field_analytic(A,phi,theta,1,coeffs[0],coeffs[1],coeffs[2],coeffs[3],coeffs[4],coeffs[5],coeffs[6],coeffs[7])
# print(coeffs)
# bfield_plot(long,lat,results[0],results[1],results[2])
def derivative_lpmv(m,n,t,dt=1e-6):
    x1 = np.cos(t+dt)
    x2 = np.cos(t-dt)
    f1 = special.lpmv(m,n,x1)
    f2 = special.lpmv(m,n,x2)
    return (f2-f1)/(2*dt)
def magnetic_field_analytic(r, theta, phi, a, g_coeff, h_coeff):
    # Initialize the magnetic field components
    B_r = 0
    B_theta = 0
    B_phi = 0
    # Calculate the field components
    for n in range(1, len(g_coeff)):  # n = 1 for dipole, n = 2 for quadrupole
        for m in range(0, n+1):
            # Radial component
            B_r += (n + 1) * (a/r)**(n+2) * (g_coeff[n][m] * np.cos(m*phi) + h_coeff[n][m] * np.sin(m*phi)) * special.lpmv(m, n, np.cos(theta))
            # Theta component
            B_theta += -(a/r)**(n+2) * (g_coeff[n][m] * np.cos(m*phi) + h_coeff[n][m] * np.sin(m*phi)) * derivative_lpmv(m,n,theta)  
            # Phi component
            B_phi += 1/(np.sin(theta)) * (a/r)**(n+2) * m * (g_coeff[n][m] * np.sin(m*phi) - h_coeff[n][m] * np.cos(m*phi)) * special.lpmv(m, n, np.cos(theta))
    return B_r, B_theta, B_phi

# Bx,By,Bz = magnetic_field_analytic(0.7,np.pi/4,np.pi/4,0.725,10,0,0,0,0,0,0,0,0,0)
# print(Bx,By,Bz)

initial_guess_g = [
    [0.0,0.0,0.0],
    [20.0,-3.0,0.0],
    [0.0,-10.0,0.0]
]
initial_guess_h = [
    [0.0,0.0,0.0],
    [0.0,0.0,0.0],
    [0.0,0.0,0.0]
]

Bx,By,Bz = magnetic_field_analytic(r,theta,phi,1740/2439.7, initial_guess_g,initial_guess_h)
# Bx,By,Bz = bfield_spherical_to_cartesian(Bx,By,Bz,theta,phi)
b_magnitude = np.sqrt(np.array(Bx)**2 + np.array(By)**2 + np.array(Bz)**2)
interp_num = 110 # number of points = square this number
Bx_interp = bfield_interpolation(long,lat,Bx,interp_num)
By_interp = bfield_interpolation(long,lat,By,interp_num)
Bz_interp = bfield_interpolation(long,lat,Bz,interp_num)
B_interp = bfield_interpolation(long,lat,b_magnitude,interp_num)

subplot_positions = [[0,0],[0,1],[1,0],[1,1]]
# plotting 3d plot of all points and the field vector
fig,ax = plt.subplots(2,2,figsize=(10,7),subplot_kw={"projection": "mollweide"})
# fig.suptitle(r'$P_{ram} = $' + f'{pram:.3f} (nPa), ' + r'$R_{SS} = $' + f'{model.rss:.3f} '+r'($R_{M}$)')
# coordinates = ax[0,0].scatter(long,lat,c=b_magnitude,cmap='seismic')
# coordinates_x=ax[0,1].scatter(long,lat,c=Bx,cmap='seismic')
# coordinates_y=ax[1,0].scatter(long,lat,c=By,cmap='seismic')
# coordinates_z=ax[1,1].scatter(long,lat,c=Bz,cmap='seismic')
coordinates = ax[0,0].contourf(B_interp[0],B_interp[1],B_interp[2],cmap='seismic',levels=80,extend='both')
coordinates_x = ax[0,1].contourf(Bx_interp[0],Bx_interp[1],Bx_interp[2],cmap='seismic',levels=80,extend='both')
coordinates_y = ax[1,0].contourf(By_interp[0],By_interp[1],By_interp[2],cmap='seismic',levels=80,extend='both')
coordinates_z = ax[1,1].contourf(Bz_interp[0],Bz_interp[1],Bz_interp[2],cmap='seismic',levels=80,extend='both')
# fig.colorbar(coordinates,label=r'$|B|$ (nT)',orientation='horizontal',ax=ax[0,0])
# fig.colorbar(coordinates_x,label=r'$B_x$ (nT)',orientation='horizontal',ax=ax[0,1])
# fig.colorbar(coordinates_y,label=r'$B_y$ (nT)',orientation='horizontal',ax=ax[1,0],format=lambda x, _: f"{x:.2f}")
# fig.colorbar(coordinates_z,label=r'$B_z$ (nT)',orientation='horizontal',ax=ax[1,1])
fig.colorbar(coordinates,label=r'$|B|$ (nT)',orientation='horizontal',ax=ax[0,0])
fig.colorbar(coordinates_x,label=r'$B_r$ (nT)',orientation='horizontal',ax=ax[0,1])
fig.colorbar(coordinates_y,label=r'$B_{\theta}$ (nT)',orientation='horizontal',ax=ax[1,0],format=lambda x, _: f"{x:.2f}")
fig.colorbar(coordinates_z,label=r'$B_{\phi}$ (nT)',orientation='horizontal',ax=ax[1,1])
for i in subplot_positions:
    ax[i[0],i[1]].grid(alpha=0.5)
fig.tight_layout()
plt.show()


# craft_num = 1
# df1 = prelim_data(pd.read_csv(f'mercury_data_1_clean.csv'))
# begintime = '1979-05-25 00:00:00'
# endtime = '1979-06-01 23:59:59'
# df1_select = time_select(df1, begintime, endtime)
# bfield_timeseries(df1_select['Pram'])












'''
# map the desired spherical surface
N = 1000
cords = generate_sphere_points(1740/2400, N)
X = cords[:,0]
Y = cords[:,1]
Z = cords[:,2] - 479/2400
R, phi, theta = cartesian_to_spherical(X, Y, Z)
lat, long = spherical_to_latlon(R, phi, theta)

# Calculating the B-field using the kt17_bfield function
pram=35
model = KT17_python(Pram=pram)
Bx, By, Bz = model.kt17_bfield(N, X, Y, Z)
# Calculate the B-field magnitude for each point
b_magnitude = np.sqrt(np.array(Bx)**2 + np.array(By)**2 + np.array(Bz)**2)

interp_num = 100 #number of points = square this number
Bx_interp = bfield_interpolation(long,lat,Bx,interp_num)
By_interp = bfield_interpolation(long,lat,By,interp_num)
Bz_interp = bfield_interpolation(long,lat,Bz,interp_num)
B_interp = bfield_interpolation(long,lat,b_magnitude,interp_num)

subplot_positions = [[0,0],[0,1],[1,0],[1,1]]
# plotting 3d plot of all points and the field vector
fig,ax = plt.subplots(2,2,figsize=(10,7),subplot_kw={"projection": "mollweide"})
fig.suptitle(r'$P_{ram} = $' + f'{pram} (nPa)')
# coordinates = ax[0,0].scatter(long,lat,c=b_magnitude,cmap='seismic')
# coordinates_x=ax[0,1].scatter(long,lat,c=Bx,cmap='seismic')
# coordinates_y=ax[1,0].scatter(long,lat,c=By,cmap='seismic')
# coordinates_z=ax[1,1].scatter(long,lat,c=Bz,cmap='seismic')
coordinates = ax[0,0].contourf(B_interp[0],B_interp[1],B_interp[2],cmap='plasma',levels=80,extend='both')
coordinates_x = ax[0,1].contourf(Bx_interp[0],Bx_interp[1],Bx_interp[2],cmap='plasma',levels=80,extend='both')
coordinates_y = ax[1,0].contourf(By_interp[0],By_interp[1],By_interp[2],cmap='plasma',levels=80,extend='both')
coordinates_z = ax[1,1].contourf(Bz_interp[0],Bz_interp[1],Bz_interp[2],cmap='plasma',levels=80,extend='both')
fig.colorbar(coordinates,label=r'$|B|$ (nT)',orientation='horizontal',ax=ax[0,0])
fig.colorbar(coordinates_x,label=r'$B_x$ (nT)',orientation='horizontal',ax=ax[0,1])
fig.colorbar(coordinates_y,label=r'$B_y$ (nT)',orientation='horizontal',ax=ax[1,0],format=lambda x, _: f"{x:.2f}")
fig.colorbar(coordinates_z,label=r'$B_z$ (nT)',orientation='horizontal',ax=ax[1,1])
for i in subplot_positions:
    ax[i[0],i[1]].grid(alpha=0.5)
fig.tight_layout()
plt.show()
'''

'''
# Filter out points for the cross-section at z=0
tolerance = 0.05  # Tolerance for considering points as part of the z=0 cross-section
cross_section_indices = np.where(np.abs(Z) < tolerance)
x_cross_section = X[cross_section_indices]
y_cross_section = Y[cross_section_indices]
b_magnitude_cross_section = b_magnitude[cross_section_indices]
'''

'''
# Plotting the heatmap
plt.figure(figsize=(8, 6))
plt.scatter(x_cross_section, y_cross_section, c=b_magnitude_cross_section, cmap='viridis')
plt.colorbar(label='B-field Magnitude')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Heatmap of B-field Magnitude at Cross-Section z=0')
plt.grid(True)
plt.show()
'''
'''
# plotting 3d plot of all points and the field vector
fig,ax = plt.subplots(1,1,figsize=(10,7),subplot_kw={"projection": "3d"})
ax.grid()
ax.set_xlabel('x '+r'$(R_M)$')
ax.set_ylabel('y '+r'$(R_M)$')
ax.set_zlabel('z '+r'$(R_M)$')
coordinates = ax.scatter(X,Y,Z,c=b_magnitude,cmap='viridis')
# directions = ax.quiver(X,Y,Z,Bx,By,Bz,length=0.1,normalize=True,color='grey')
fig.colorbar(coordinates,label='|B| (nT)')
plt.show()
'''

