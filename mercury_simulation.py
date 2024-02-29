import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KT17_python_smz16 import KT17_python
from scipy import interpolate
from scipy import optimize
from scipy import special


class Mercury_B_induced_sim:
    def __init__(self,n=1000,surface_r=1740) -> None: # initiate the surface required
        # map the desired spherical surface
        self.N = n
        self.R_M = 2439.7
        cords = self.generate_sphere_points(surface_r/self.R_M, self.N)
        self.X = cords[:,0]
        self.Y = cords[:,1]
        self.Z = cords[:,2] #- 479/self.R_M # offsetting the magnetic dipole centre from planet's centre since they're different
        # changing it to lattitude and longtitude to plot in hte mollweide projection 
        self.R, self.theta, self.phi = self.cartesian_to_spherical(self.X, self.Y, self.Z)
        self.long, self.lat = self.spherical_to_latlon(self.R, self.theta, self.phi)
        self.A = surface_r/self.R_M
        self._cartesian = False
        pass

    def generate_sphere_points(self,radius, num_points, spherical=False):
        """
        Generate points on the surface of a sphere using the Fibonacci lattice method. (ChatGPT)

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

            if spherical==True:
                # Convert Cartesian coordinates to spherical coordinates
                radius, azimuthal_angle, polar_angle = self.cartesian_to_spherical(x, y, z)
                points.append((np.degrees(polar_angle), np.degrees(azimuthal_angle)))
            else:
                points.append((radius * x, radius * y, radius * z))
        return np.array(points)
    
    def cartesian_to_spherical(self,x, y, z):
        """ coordinate transformation from cartesian to spherical coordinates """
        radius = np.sqrt(x**2 + y**2 + z**2)
        azimuthal_angle = np.arccos(z / radius)
        polar_angle = np.arctan2(y, x)
        return radius, azimuthal_angle, polar_angle
    def spherical_to_latlon(self,radius, azimuthal_angle, polar_angle):
        """ express points on surface in lattitude and longtitude from spherical coordinates """
        latitude = np.pi / 2 - azimuthal_angle
        longitude = polar_angle
        return longitude, latitude
    def bfield_cartesian_to_spherical(self,bx,by,bz,t,p):
        """ transform magnetic field data from cartesian to spherical coordinates """
        br = bx*np.sin(t)*np.cos(p) + by*np.sin(t)*np.sin(p) + bz*np.cos(t)
        btheta = bx*np.cos(t)*np.cos(p) + by*np.cos(t)*np.sin(p) - bz*np.sin(t)
        bphi = bx*(-np.sin(p)) + by*np.cos(p)
        return br, btheta, bphi
    def bfield_spherical_to_cartesian(self,br,btheta,bphi,t,p):
        """ transform magnetic field data from spherical to cartesian coordinates """
        bx = br*np.sin(t)*np.cos(p) + btheta*np.cos(t)*np.cos(p) - bphi*np.sin(p)
        by = br*np.sin(t)*np.sin(p) + btheta*np.cos(t)*np.sin(p) + bphi*np.cos(p)
        bz = br*np.cos(t) - btheta*np.sin(t)
        return bx,by,bz
    def spherical_to_cartesian(self,r,t,p):
        """ coordinate transformation from spherical to cartesian coordinates """
        x = r*np.sin(t)*np.cos(p)
        y = r*np.sin(t)*np.sin(p)
        z = r*np.cos(t)
        return x,y,z
    def bfield_interpolation(self,xval,yval,Bval,num_interp):
        """ interpolation of the bfield data on the surface to get more data between points and plot """
        # for plotting
        grid_x, grid_y = np.meshgrid(np.linspace(min(xval), max(xval), num_interp), np.linspace(min(yval), max(yval), num_interp))
        grid_z = interpolate.griddata((xval,yval),Bval,(grid_x,grid_y),method='cubic')
        return grid_x,grid_y,grid_z
    def bfield_inducing(self,ram_pres_series,cartesian=False):
        """ generate the inducing magnetic field at Mercury using the KT17 model, by feeding in solar wind ram pressure """
        # start looping through a timeseries data of Pram
        ram_pres_series = list(np.array(ram_pres_series)*1e9)
        self.bfield_list = []
        for i in range(len(ram_pres_series)):
            pram=ram_pres_series[i] # in nPa
            # Calculating the B-field using the kt17_bfield function
            model = KT17_python(Pram=pram)
            Bx, By, Bz = model.kt17_bfield(self.N, self.X, self.Y, self.Z)
            b_magnitude = np.sqrt(np.array(Bx)**2 + np.array(By)**2 + np.array(Bz)**2)
            # Calculate the B-field magnitude for each point
            if cartesian == False:
                Bx, By, Bz = self.bfield_cartesian_to_spherical(Bx,By,Bz,self.theta,self.phi)# label bx,by,bz but are actually br, btheta, bphi
                b_magnitude = np.sqrt(np.array(Bx)**2 + np.array(By)**2 + np.array(Bz)**2)
            self.bfield_list.append([Bx,By,Bz,b_magnitude,pram,model.rss])
        return self
    
    def delta_bfield_inducing(self,ram_pres_series,cartesian=False):
        """ Finding the change in magnetic field, the inducing field """
        # start looping through a timeseries data of Pram
        ram_pres_series = list(np.array(ram_pres_series)*1e9)
        self.bfield_list = []
        bx_list = []
        by_list = []
        bz_list = []
        pram_list = []
        rss_list = []
        for i in range(len(ram_pres_series)):
            pram=ram_pres_series[i] # in nPa
            # Calculating the B-field using the kt17_bfield function
            model = KT17_python(Pram=pram)
            Bx, By, Bz = model.kt17_bfield(self.N, self.X, self.Y, self.Z)
            # Calculate the B-field magnitude for each point
            if cartesian == False:
                Bx, By, Bz = self.bfield_cartesian_to_spherical(Bx,By,Bz,self.theta,self.phi) # label bx,by,bz but are actually br, btheta, bphi
            self.bfield_list.append([Bx,By,Bz])
            bx_list.append(Bx)
            by_list.append(By)
            bz_list.append(Bz)
            pram_list.append(pram)
            rss_list.append(model.rss)
        # method 1: subtract mean
        # dbx = np.array(bx_list) - np.mean(bx_list)
        # dby = np.array(by_list) - np.mean(by_list)
        # dbz = np.array(bz_list) - np.mean(bz_list)
        # method 2: subtract first snapshot
        dbx = np.array(bx_list) - bx_list[0]
        dby = np.array(by_list) - by_list[0]
        dbz = np.array(bz_list) - bz_list[0]
        new_bfield_list=[]
        for i in range(len(dbx)):
            new_bfield_list.append([dbx[i],dby[i],dbz[i],np.sqrt(dbx[i]**2+dby[i]**2+dbz[i]**2),pram_list[i],rss_list[i]])
        self.bfield_list = new_bfield_list
        return self

    def bfield_plot(self,convert_to_cartesian=False,label='spherical'):
        """ plotting any magnetic field data on a surface """
        pram_start = self.bfield_list[0][4]
        for i in range(len(self.bfield_list)):
            Bx, By, Bz, b_magnitude, pram, rss = self.bfield_list[i]
            if convert_to_cartesian == True:
                Bx, By, Bz = self.bfield_spherical_to_cartesian(Bx,By,Bz,self.theta,self.phi)
            interp_num = 110 # number of points = square this number
            Bx_interp = self.bfield_interpolation(self.long,self.lat,Bx,interp_num)
            By_interp = self.bfield_interpolation(self.long,self.lat,By,interp_num)
            Bz_interp = self.bfield_interpolation(self.long,self.lat,Bz,interp_num)
            B_interp = self.bfield_interpolation(self.long,self.lat,b_magnitude,interp_num)

            subplot_positions = [[0,0],[0,1],[1,0],[1,1]]
            # plotting 3d plot of all points and the field vector
            fig,ax = plt.subplots(2,2,figsize=(10,7),subplot_kw={"projection": "mollweide"})
            fig.suptitle(r'$\delta P_{ram} = $' + f'{pram-pram_start:.3f} (nPa), ' + r'$R_{SS} = $' + f'{rss:.3f} '+r'($R_{M}$)')
            # coordinates = ax[0,0].scatter(long,lat,c=b_magnitude,cmap='seismic')
            # coordinates_x=ax[0,1].scatter(long,lat,c=Bx,cmap='seismic')
            # coordinates_y=ax[1,0].scatter(long,lat,c=By,cmap='seismic')
            # coordinates_z=ax[1,1].scatter(long,lat,c=Bz,cmap='seismic')
            coordinates = ax[0,0].contourf(B_interp[0],B_interp[1],B_interp[2],cmap='seismic',levels=80,extend='both')
            coordinates_x = ax[0,1].contourf(Bx_interp[0],Bx_interp[1],Bx_interp[2],cmap='seismic',levels=80,extend='both')
            coordinates_y = ax[1,0].contourf(By_interp[0],By_interp[1],By_interp[2],cmap='seismic',levels=80,extend='both')
            coordinates_z = ax[1,1].contourf(Bz_interp[0],Bz_interp[1],Bz_interp[2],cmap='seismic',levels=80,extend='both')
            if label == 'cartesian':
                fig.colorbar(coordinates,label=r'$|B|$ (nT)',orientation='horizontal',ax=ax[0,0])
                fig.colorbar(coordinates_x,label=r'$B_x$ (nT)',orientation='horizontal',ax=ax[0,1])
                fig.colorbar(coordinates_y,label=r'$B_y$ (nT)',orientation='horizontal',ax=ax[1,0])#,format=lambda x, _: f"{x:.2f}")
                fig.colorbar(coordinates_z,label=r'$B_z$ (nT)',orientation='horizontal',ax=ax[1,1])
            elif label == 'spherical':
                fig.colorbar(coordinates,label=r'$|B|$ (nT)',orientation='horizontal',ax=ax[0,0])
                fig.colorbar(coordinates_x,label=r'$B_r$ (nT)',orientation='horizontal',ax=ax[0,1])
                fig.colorbar(coordinates_y,label=r'$B_{\theta}$ (nT)',orientation='horizontal',ax=ax[1,0])#,format=lambda x, _: f"{x:.2f}")
                fig.colorbar(coordinates_z,label=r'$B_{\phi}$ (nT)',orientation='horizontal',ax=ax[1,1])
            elif label == 'spherical_delta':
                fig.colorbar(coordinates,label=r'$|\delta B|$ (nT)',orientation='horizontal',ax=ax[0,0])
                fig.colorbar(coordinates_x,label=r'$\delta B_r$ (nT)',orientation='horizontal',ax=ax[0,1])
                fig.colorbar(coordinates_y,label=r'$\delta B_{\theta}$ (nT)',orientation='horizontal',ax=ax[1,0])#,format=lambda x, _: f"{x:.2f}")
                fig.colorbar(coordinates_z,label=r'$\delta B_{\phi}$ (nT)',orientation='horizontal',ax=ax[1,1])
            
            for i in subplot_positions:
                ax[i[0],i[1]].grid(alpha=0.5)
            fig.tight_layout()
        plt.show()
        return self
    # minimise the analytical b field to get gauss coefficients
    def derivative_lpmv(self,m,n,t,dt=1e-6):
        x1 = np.cos(t+dt)
        x2 = np.cos(t-dt)
        f1 = special.lpmv(m,n,x1)
        f2 = special.lpmv(m,n,x2)
        return (f2-f1)/(2*dt)
 
    def magnetic_field_analytic(self, r, theta, phi, a, g_coeff, h_coeff, fixed_param=True,induced=False):
        """ Magnetic Field Function """
        # Initialize the magnetic field components
        B_r = 0
        B_theta = 0
        B_phi = 0
        # fix parameters (comment out section if not wanted)
        if fixed_param == True:
            g01 = g_coeff[1][0]
            g11 = g_coeff[1][1]
            g12 = g_coeff[2][1]
            if len(g_coeff) == 4:
                g03 = g_coeff[3][0]
            h_coeff = np.zeros((int(len(h_coeff)),int(len(h_coeff))))
            g_coeff = np.zeros((int(len(g_coeff)),int(len(g_coeff))))
            g_coeff[1][0] = g01
            g_coeff[1][1] = g11
            g_coeff[2][1] = g12
            if len(g_coeff) == 4:
                g_coeff[3][0] = g03
        # Calculate the field components
        if induced == False:
            for n in range(1, len(g_coeff)):  # n = 1 for dipole, n = 2 for quadrupole
                for m in range(0, n+1):
                    # Radial component
                    B_r += -(n) * (r/a)**(n-1) * (g_coeff[n][m] * np.cos(m*phi) + h_coeff[n][m] * np.sin(m*phi)) * special.lpmv(m, n, np.cos(theta))
                    # Theta component
                    B_theta += (r/a)**(n-1) * (g_coeff[n][m] * np.cos(m*phi) + h_coeff[n][m] * np.sin(m*phi)) * self.derivative_lpmv(m,n,theta)  
                    # Phi component
                    B_phi += 1/(np.sin(theta)) * (r/a)**(n-1) * m * (g_coeff[n][m] * np.sin(m*phi) + h_coeff[n][m] * np.cos(m*phi)) * special.lpmv(m, n, np.cos(theta))
        else:
            for n in range(1, len(g_coeff)):  # n = 1 for dipole, n = 2 for quadrupole
                for m in range(0, n+1):
                    # Radial component
                    B_r += (n+1) * (a/r)**(n+2) * (g_coeff[n][m] * np.cos(m*phi) + h_coeff[n][m] * np.sin(m*phi)) * special.lpmv(m, n, np.cos(theta))
                    # Theta component
                    B_theta += (a/r)**(n+2) * (g_coeff[n][m] * np.cos(m*phi) + h_coeff[n][m] * np.sin(m*phi)) * self.derivative_lpmv(m,n,theta)  
                    # Phi component
                    B_phi += 1/(np.sin(theta)) * (a/r)**(n+2) * m * (g_coeff[n][m] * np.sin(m*phi) - h_coeff[n][m] * np.cos(m*phi)) * special.lpmv(m, n, np.cos(theta))
        return B_r, B_theta, B_phi
    
    def bfield_rms_residual(self,coeffs, r, t, p, br_actual, bt_actual, bp_actual):
        """ calculate the root-mean-square value of the magnetic field difference according to model and actual data """
        # reshape
        g_coeff, h_coeff = np.array(coeffs).reshape((2,int(len(coeffs)/2)))[0], np.array(coeffs).reshape((2,int(len(coeffs)/2)))[1]
        g_coeff = np.array(g_coeff).reshape((int(np.sqrt(len(g_coeff))),int(np.sqrt(len(g_coeff)))))
        h_coeff = np.array(h_coeff).reshape((int(np.sqrt(len(h_coeff))),int(np.sqrt(len(h_coeff)))))
        # model
        br_model, bt_model, bp_model = self.magnetic_field_analytic(r,t,p,self.A,g_coeff,h_coeff)
        # return np.sqrt(np.sum([(br_model[i]-br_actual[i])**2 + (bt_model[i]-bt_actual[i])**2 + (bp_model[i]-bp_actual[i])**2 for i in range(len(bp_actual))])/3/len(br_actual))
        return np.sqrt(np.sum(((br_model-br_actual)**2 + (bt_model-bt_actual)**2 + (bp_model-bp_actual)**2))/len(br_actual))
    
    def bfield_rms_residual_plot(self,g12=0):
        """ Plotting the minimisng function (optional) """
        g01s = np.linspace(-10,10,100)
        g11s = np.linspace(-10,10,100)
        g12 = g12
        rms_vals=np.zeros((len(g01s),len(g11s)))
        for i in range(len(self.bfield_list)):
            bra, bta, bpa = self.bfield_list[i][0],self.bfield_list[i][1],self.bfield_list[i][2]
            for j in range(len(g01s)):
                for k in range(len(g11s)):
                    rms_vals[k][j] = self.bfield_rms_residual([0,0,0,g01s[j],g11s[k],0,0,g12,0,0,0,0,0,0,0,0,0,0],self.R,self.theta,self.phi,bra,bta,bpa)
            plt.figure()
            plt.title(r'$g_{12} = $' + f'{g12}')
            plt.xlabel(r'$g_{01}$')
            plt.ylabel(r'$g_{11}$')
            plt.imshow(rms_vals,extent=[-10,10,-10,10],origin='lower',cmap='plasma')
            plt.colorbar(label=r'$B_{RMS}$')
        plt.show()
    
    def bfield_coefficient_minimisation(self,init_guess):
        """ 
        minimise the root-mean-square of the magnetic field difference to find the gauss coefficient that corresponds to the data, 
        can use this function to feed bfield to bfield_plot() to plot 
        """
        self.minimisation_result = []
        minimised_field_list = []
        residual_list = []
        init_guess = np.array(init_guess).flatten()
        for i in range(len(self.bfield_list)):
            bra, bta, bpa = self.bfield_list[i][0],self.bfield_list[i][1],self.bfield_list[i][2] # current bfield data
            # minimise
            minimise = optimize.minimize(self.bfield_rms_residual,x0=init_guess,args=(self.R,self.theta,self.phi,bra,bta,bpa))
            # minimise = optimize.basinhopping(self.bfield_rms_residual,x0=init_guess,minimizer_kwargs={"args":(self.R,self.theta,self.phi,bra,bta,bpa)})
            coeffs = minimise.x # minimised gauss coefficient for the magnetosphere
            residual_list.append(self.bfield_rms_residual(coeffs,self.R,self.theta,self.phi,bra,bta,bpa))
            # reshape
            g_coeff, h_coeff = np.array(coeffs).reshape((2,int(len(coeffs)/2)))[0], np.array(coeffs).reshape((2,int(len(coeffs)/2)))[1]
            g_coeff = list(np.array(g_coeff).reshape((int(np.sqrt(len(g_coeff))),int(np.sqrt(len(g_coeff))))))
            h_coeff = list(np.array(h_coeff).reshape((int(np.sqrt(len(h_coeff))),int(np.sqrt(len(h_coeff))))))
            self.minimisation_result.append([g_coeff,h_coeff])
            # recalculate the magnetic field according to the minimised gauss coefficients values
            minimised_field = self.magnetic_field_analytic(self.R,self.theta,self.phi,self.A,g_coeff,h_coeff)
            minimised_field_mag = np.sqrt(np.array(minimised_field[0])**2 + np.array(minimised_field[1])**2 + np.array(minimised_field[2])**2)
            # save the field that the minimised gauss coefficients give
            minimised_field_list.append([minimised_field[0],minimised_field[1],minimised_field[2],minimised_field_mag,self.bfield_list[i][4],self.bfield_list[i][5]])
        self.bfield_list = minimised_field_list
        print(f'Mean Residual = {np.array(residual_list).mean()} (nT)')
        # print(f'Residuals = {np.array(residual_list)} (nT)')
        return self
    
    def inducing_to_induced_coeff(self,gh_coeff,ratio=False):
        """ Transfer Function """
        gh_coeff_induced = []
        induced_ratio = []
        for n in range(len(gh_coeff)):
            for m in range(len(gh_coeff[n])):
                # transfer function, under the assumption of steep conductivity change
                induced_coeff = n/(n+1) * (self.A)**(2*n+1) * gh_coeff[n][m] 
                gh_coeff_induced.append(induced_coeff)
                induced_ratio.append(abs(n/(n+1) * (self.A)**(2*n+1)))
        gh_coeff_induced = np.array(gh_coeff_induced).reshape((int(len(gh_coeff)),int(len(gh_coeff))))
        induced_ratio = np.array(induced_ratio).reshape((int(len(gh_coeff)),int(len(gh_coeff))))
        # print(induced_ratio)
        if ratio == False:
            return gh_coeff_induced
        else:
            return gh_coeff_induced, induced_ratio
    
    def bfield_induced(self):
        """ Substitute induced coefficient back into the magnetic field """
        self.induced_coeff_list = []
        self.bfield_induced_list = []
        for i in range(len(self.minimisation_result)):
            g_coeff_induced, h_coeff_induced = self.inducing_to_induced_coeff(self.minimisation_result[i][0]), self.inducing_to_induced_coeff(self.minimisation_result[i][1])
            self.induced_coeff_list.append([g_coeff_induced,h_coeff_induced])
            br, bt, bp = self.magnetic_field_analytic(self.R,self.theta,self.phi,self.A,g_coeff_induced,h_coeff_induced,induced=True)
            self.bfield_induced_list.append([br,bt,bp,np.sqrt(np.array(bt)**2+np.array(br)**2+np.array(bp)**2),self.bfield_list[i][4],self.bfield_list[i][5]])
        self.bfield_list = self.bfield_induced_list
        # self.Z = self.Z - 479/self.R_M # offsetting the magnetic dipole centre from planet's centre since they're different
        # ratioes = self.inducing_to_induced_coeff(self.minimisation_result[i][0],ratio=True)[1]
        # print(f'Induced Ratio: g01 = {ratioes[1][0]:.3f}, g11 = {ratioes[1][1]:.3f}, g12 = {ratioes[2][1]:.3f}, g03 = {ratioes[3][0]:.3f}')
        return self
    
    def spacecraft_simulation(self,coordinates,plotting=True):
        """ Simulate the spacecraft trajectory and its measurement """
        self.bfield_data_log = []
        for t in range(len(self.induced_coeff_list)):
            br_inducing, bt_inducing, bp_inducing = self.magnetic_field_analytic(coordinates[t][0],coordinates[t][1],coordinates[t][2],self.A,self.minimisation_result[t][0],self.minimisation_result[t][1])
            br_induced, bt_induced, bp_induced = self.magnetic_field_analytic(coordinates[t][0],coordinates[t][1],coordinates[t][2],self.A,self.induced_coeff_list[t][0],self.induced_coeff_list[t][1],induced=True)
            self.bfield_data_log.append([br_induced,bt_induced,bp_induced,np.sqrt(br_induced**2+bt_induced**2+bp_induced**2),np.sqrt(br_induced**2+bt_induced**2+bp_induced**2)/np.sqrt(br_inducing**2+bt_inducing**2+bp_inducing**2)])
        print(np.array(self.bfield_data_log)[:,3])
        if plotting == True:
            fig, ax = plt.subplots(1,1,figsize=(7,7),subplot_kw={"projection":"3d"})
            ax.set_xlabel(r'x $(R_M)$')
            ax.set_ylabel(r'y $(R_M)$')
            ax.set_zlabel(r'z $(R_M)$')
            # ax.scatter([0],[0],[0],color='grey',s=100,label='Mercury')
            # p = np.linspace(0, 2*np.pi, 100)
            # t = np.linspace(0, np.pi, 100)
            # sphere_x = 1 * np.outer(np.cos(p), np.sin(t))
            # sphere_y = 1 * np.outer(np.sin(p), np.sin(t))
            # sphere_z = 1 * np.outer(np.ones(np.size(p)), np.cos(t))
            # ax.plot_surface(sphere_x,sphere_y,sphere_z,color='grey')
            sphere = self.generate_sphere_points(0.9,1000)
            sphere_x,sphere_y,sphere_z = sphere[:,0],sphere[:,1],sphere[:,2]
            ax.scatter(sphere_x,sphere_y,sphere_z,s=100,color='grey',label='Mercury',zorder=1)
            coord_x, coord_y, coord_z = self.spherical_to_cartesian(coordinates[:,0],coordinates[:,1],coordinates[:,2])
            ax.plot(coord_x,coord_y,coord_z,'-',color='black',label='Spacecraft',zorder=10)
            ax.set_xlim(-4,4)
            ax.set_ylim(-4,4)
            ax.set_zlim(-4,4)
            fig.legend()
            # fig.tight_layout()
            plt.show()
        return self
    
    # def bfield_plot_3D(self):
    #     """ Plotting the magnetic field lines in 3D """
    #     cordinates = self.generate_sphere_points(1, 50)
    #     X = cordinates[:,0]
    #     Y = cordinates[:,1]
    #     Z = cordinates[:,2] - 479/self.R_M # offsetting the magnetic dipole centre from planet's centre since they're different
    #     # changing it to lattitude and longtitude to plot in hte mollweide projection 
    #     R, theta, phi = self.cartesian_to_spherical(X, Y, Z)
    #     coordinates = np.column_stack((R,theta,phi))
    #     for i in range(1,len(self.induced_coeff_list)-1):
    #         field_lines = []
    #         for j in range(len(coordinates)):
    #             field_line = []
    #             field_line.append([coordinates[j][0],coordinates[j][1],coordinates[j][2]])
    #             for k in range(100):
    #                 alpha = 1
    #                 br, bt, bp = self.magnetic_field_analytic(field_line[-1][0],field_line[-1][1],field_line[-1][2],self.A,self.induced_coeff_list[i][0],self.induced_coeff_list[i][1])
    #                 # new point coordinates in spherical, need to change to cartesian for plotting
    #                 new_r = field_line[-1][0]+np.sqrt(alpha**2 - (field_line[-1][0]*bt/br)**2 - (field_line[-1][0]*bp/br)**2),
    #                 new_theta = field_line[-1][1]+bt/br,
    #                 new_phi = field_line[-1][2]+bp/br
    #                 next_point = [new_r*np.sin(new_phi)*np.cos(new_theta),new_r*np.sin(new_phi)*np.sin(new_theta),new_r*np.cos(new_phi)]                
    #                 field_line.append(next_point)
    #             field_lines.append(field_line)
    #         fig ,ax = plt.subplots(2,2,figsize=(10,7),subplot_kw={"projection": "3d"})
    #         for lines in field_lines:
    #             lines_x = lines[:,0]
    #             lines_y = lines[:,1]
    #             lines_z = lines[:,2]
    #             ax.plot(lines_x,lines_y,lines_z,'-',color='black')
    #         plt.show()
    #     return self
            
  