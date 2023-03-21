# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:46:58 2022

@author: Jesper Nielsen
"""
import numpy as np
from chemistry import Abundance as ab
from scipy import interpolate
import pandas as pd
import scipy.integrate as integrate
import matplotlib.pyplot as plt

class IntegrateJoanna:
    
    def __init__(self,theta,data,dt,init_abun,
                 starttimes,t0=0,t_end_full=3*10**6,
                 iso="bitsch",free_iron = False,carbon_grain=True,
                 store_full = False,evap=True,constant_flux=False,viscous_heating=False):
        '''
        Class for simulating growth of planets through pebble accretion
        Takes into account abundance and masses of stars and outputs
        compositions and final masses as well as positions of planets

        Parameters
        ----------
        theta : array
            Input disc/model parameters.
        data : array
            Initial positions and masses of the embryos. (N_bodies, N_star)
        dt : float
            timestep.
        init_abun : array
            initial abundances of all stars.
        t0 : float
            starting point of the integration.
        t_end : float
            end of integration.
        starttimes : array
            Sets the time when the embryos should start growing.
        iso : str, optional
            Set how I should calcualte the isolation mass.
            Either from Berts paper or from gap mass found by kanagawa.
            The default is "bitsch".
        free_iron : bool, optional
            Sets if I want free iron of fayalite and magnetite.
            The default is False.
        store_full : bool, optional
            Sets if I want to store all timesteps. The default is False.
        evap : bool, optional
            Sets if I consider evaporation of species during pebble accretion
        Raises
        ------
        ValueError
            If wrong iso type is set.

        Returns
        -------
        None.

        '''
        self.St,\
        self.Z,\
        self.alpha,\
        self.M_star,\
        self.zeta,\
        self.beta,\
        self.kappa,\
        self.delta,\
        self.v_f = theta
        
        self.data = data
        self.N_star = theta.shape[1]
        self.store_full = store_full
        self.starttimes = starttimes
                
        self.iso = iso
        
        self.evap = evap
        
        self.constant_flux = constant_flux
        
        self.viscous_heating = viscous_heating
        
        iso_names = ["bitsch","gap"]
        
        if iso not in iso_names:
            
            raise ValueError("Wrong iso name")
        
        self.Mps2AUpYr = 2.108*10**-4
        self.MSun2ME = 1.9891*10**30/(5.9722*10**24)
        self.G = 1.184*10**-4 #AU^3 M_E^-1 yr^-2
        self.u = 1.66*10**-27 # kg
        self.kg2ME = 1.674*10**-25 
        
        self.alpha_3 = np.repeat(10**-3,self.N_star)
        self.alpha_v = np.repeat(1*10**-4,self.N_star)
        self.eps_p = 0.5
        self.eps_d = self.eps_p/10
        
        self.ab_class = ab(init_abun,free_iron,carbon_grain)
        
        #Time stuff
        #Ended up not mattering too much but have logarithmic timesteps up until
        #a maximum
        
        self.disc_lifetimes = t_end_full
        self.t_end = np.max(t_end_full)
        self.max_dt=dt
        #times = np.logspace(0,np.log10(self.t_end),100)
        #times = times[np.insert(np.diff(times),0,0)<self.max_dt]
        #self.times = np.append(times,np.arange(np.max(times)+self.max_dt,self.t_end,self.max_dt))
        self.times = np.arange(t0,self.t_end,dt)
                
        self.N_step = len(self.times)
        
        gas_norm = 0.1
        
        if gas_norm != 0.1:
            
            print('Warning gas norm = {}'.format(gas_norm))
        
        self.M_gas = (gas_norm*(self.M_star/self.MSun2ME)**1.4)*self.MSun2ME
        self.lums = self.calc_lum(self.times)
        
        self.gamma=3/2-self.zeta
        
        #Set initial gas flux to be equal to 10 times the X-ray photoevaporation
        self.M_dotend = self.M_gphoto()
        self.M_0dot = 10*self.M_dotend
        
        #Inner edge of disc
        self.r_in = np.repeat(0.1,self.N_star) #AU
                    
        #Boltzmanns constant - used if I want to calculate pressures
        self.kb = 1.38*10**-23 #m^2 kg s^-2 K^-1
        self.sigma_sb = 2.978*10**-10
        self.mh = 1.00784*self.u #kg

        self.init_gas_mass = 0.1*(self.M_star/self.MSun2ME)**1.4
        
        #This can change - look into
        self.mu_disc = 2.3
               
        #Characteristic disc size
        self.R1 = self.get_R1()
        
        #Negative logarithmic pressure gradient
        self.chi_val = self.beta+self.zeta/2+3/2
        self.gamma = 3/2-self.zeta
        
        #Used to calculate the temperature at the RCB
        self.gamma_ad = 1.4
        self.beta_opacity = 2
        
        #Radial migration coefficient
        self.kmig = 2*(1.36+0.62*self.beta+0.43*self.zeta)
        
        #Abundance stuff
        #Need to calculate inital abundance of embryos, assume that their 
        #initial abundance reflect that of the solid material at the position
        #where they are intialised
        
        self.N_species = self.ab_class.N_species
        self.r_0 = data[:,:,1]
        self.M_0 = data[:,:,0]
        self.L_0 = self.calc_lum(self.starttimes)
        
        #If we have viscous heating, the temperature of the disc is equal to 
        #the maximum of viscous and irradtion heating
        
        #Otherwise only irradiation
        if self.viscous_heating:
            
            self.T_irr = 150*(self.M_star/self.MSun2ME)**(-1/7)*\
                self.L_0**(2/7)*\
                (self.r_0)**(-self.zeta)
                
            self.T_visc = 200*(self.M_star/self.MSun2ME)**(3/10)*\
                (self.alpha/10**-3)**(-1/5)*\
                (self.M_gdot(self.starttimes)/self.MSun2ME/10**-8)**(2/5)*\
                self.r_0**(-9/10)
            
            self.T_0 = np.max(np.stack((self.T_irr,self.T_visc)),axis=0)
            
        else:        
            
            self.T_0 = 150*(self.M_star/self.MSun2ME)**(-1/7)*self.L_0**(2/7)*(self.r_0)**(-self.zeta)
        
        self.N_bodies = self.r_0.shape[0]
        
        #Data of all chemical species used
        self.spec_data = pd.read_csv("species_data.csv",delimiter=";")
        
        #Species-specific constants needed for calculations
        #Need to extend them so they cover all bodies and all stars
        self.density = self.spec_data["density"].values
        self.density = np.repeat(self.density[np.newaxis,:],self.N_bodies,axis=0)
        self.density = np.repeat(self.density[:,np.newaxis,:],self.N_star,axis=1)

        #Initialise the solid abundances of the protoplanets
        #They do not have a gaseous component yet - set to zero
        self.init_solid = self.ab_class.M_species(self.M_0,self.T_0,gas=False)
        self.init_gas = np.zeros((self.N_bodies,self.N_star,self.N_species))
        
        if store_full:
            
            #Arrays for storage
            self.data_out = np.zeros((self.N_step+1,self.N_bodies,self.N_star,2))
            self.solids = np.zeros((self.N_step+1,self.N_bodies,self.N_star,self.N_species))
            self.gases = np.zeros((self.N_step+1,self.N_bodies,self.N_star,self.N_species))
        
    def M_gdot(self,t):
            
        gamma=3/2-self.zeta
        
        T = 150*(self.M_star/self.MSun2ME)**(-1/7)*self.calc_lum(0)**(2/7)*(self.R1)**(-self.zeta)
        c_s = self.Mps2AUpYr*np.sqrt(self.kb*T/(self.mu_disc*self.mh))
        omega = np.sqrt(self.G*self.M_star/self.R1**3)
        H = c_s/omega
        nu = self.alpha*c_s*H
        
        ts = 1/(3*(2-gamma)**2)*self.R1**2/nu
        
        exp = -(5/2-gamma)/(2-gamma)
        
        return self.M_0dot*(t/ts+1)**exp
    
    def calc_lum(self,times):
        '''
        
        Parameters
        ----------
        times : array
            array with the different timesteps .

        Returns
        -------
        lum_data : array
            Luminosities for each stars and each timestep.

        '''
        
        #Read isochrones and find each stellar mass value
        iso = pd.read_csv("mass_lum_isochrones.csv")
        isomass = iso["mass"].values        
        unique_isomass = np.unique(isomass)
        
        N_time = times.shape[0]
        
        lum_data = np.zeros((N_time,self.N_star))
        
        #Loop over each star
        for i,star in enumerate(self.M_star/self.MSun2ME):
            
            #Check for time array dimension
            if times.ndim==2:
                times_calc = times[:,i]
            else:
                times_calc = times
            
            #Find the closest mass value in isochrone and get L(t)
            diff = abs(unique_isomass-star)
            
            nearest = np.where(diff == np.min(diff))
            
            near_mass = unique_isomass[nearest]
            
            filt = isomass == near_mass
            
            iso_t = 10**iso["logt"].values[filt]
            iso_lum = 10**iso["lum"].values[filt]
            
            #Linearly interpolate - set bounds error to False to avoid nan
            #at edges - instead extrapolates
            #interp1d creates a function which can be called to get the values
            f = interpolate.interp1d(iso_t,iso_lum,bounds_error=False)
                       
            lum_data[:,i] = f(times_calc)
            
            lum_data[np.isnan(lum_data)] = f(min(iso_t))
        
        return lum_data
           
    def M_gphoto(self):
        '''
        Mass loss from X-rays as foundd by Owen et al. (2012)
        '''
        L_x = 10**(30.37+1.44*np.log10(self.M_star/self.MSun2ME))
        
        M_g = (6.25*10**(-9)*(self.M_star/self.MSun2ME)**(-0.068)*(L_x/10**30)**(1.14)).astype(float)
        
        return M_g*self.MSun2ME

    def get_R1(self):
        '''
        Calculate disc size from integrating the surface density and solving
        the analytical integral
        '''
    
        A = 39*self.kb*7.44*10**(-33)*self.alpha*150*self.calc_lum(np.array([0]))**(2/7)*(self.M_star/self.MSun2ME)**(-1/7)*self.M_gas
        
        B = 28*self.M_0dot*np.sqrt(self.G*self.M_star)*self.mu_disc*self.mh*self.kg2ME

        R_1 =(self.r_in**(13/14)+A/B)**(14/13)
        
        return R_1
    
    def integrate(self):

        y = self.data
        y_solid = self.init_solid
        y_gas = self.init_gas
        
        #If we have viscous heating, the temperature at the inner region of the disc,
        #might be high enough for all refractories to be evaporated
        #This means we cannot set the starting point of the pebble formation front
        #at a fixed value. Instead we set it to the maximum of inner edge and 
        #where the first species have condensed
        
        if self.store_full:
            
            #data_out consists of (time,body,star,data), where the data is mass and radius
            self.data_out[0] = y
            self.solids[0] = y_solid
            self.gases[0] = y_gas
        
        dt = self.times[1]-self.times[0]
        
        if not self.constant_flux:
            
            #Initialise a grid at which I calculate the initial dust mass budget
            #Need to track the remaining mass budget at several location at the same time
            #to take into account planet migration, easiest to do in a grid
            
            N_grid = 50
            r_grid = np.logspace(np.log10(self.r_in),np.log10(self.R1[0]),N_grid)
            grid_points = (r_grid[1:,:]+r_grid[:-1,:])/2
    
            r_grid_out = np.logspace(np.log10(grid_points),np.log10(self.R1[0]),1000)
            
            #r_grid_gp = np.zeros((N_grid,self.N_star))
    
            #r_grid_gp[0] = 1.5*grid_points[0]-0.5*grid_points[1]
            #r_grid_gp[1:-1] = 0.5*(grid_points[1:]+grid_points[:-1])
            #r_grid_gp[-1] = 1.5*grid_points[-1] - 0.5*grid_points[-2]
            #dr = r_grid_gp[1:] - r_grid_gp[:-1]
            
            M_dust_init = np.zeros((N_grid-1,self.N_star))
    
            for i_grid in range(N_grid-1):
                
                r_vals = r_grid_out[:,i_grid,:]
                
                if self.viscous_heating:
                    
                    T_irr_init_grid = 150*(self.M_star/self.MSun2ME)**(-1/7)*\
                        self.calc_lum(0)**(2/7)*\
                        (r_vals)**(-self.zeta)
                        
                    T_visc_init_grid = 200*(self.M_star/self.MSun2ME)**(3/10)*\
                        (self.alpha/10**-3)**(-1/5)*\
                        (self.M_0dot/self.MSun2ME/10**-8)**(2/5)*\
                        r_vals**(-9/10)
                    
                    T_init_grid = np.max(np.stack((T_irr_init_grid,T_visc_init_grid)),axis=0)
                    
                else:        
                    
                    T_init_grid = 150*(self.M_star/self.MSun2ME)**(-1/7)*self.calc_lum(np.zeros((1,self.N_star)))**(2/7)*(r_vals)**(-self.zeta)
    
                c_s_init_grid = self.Mps2AUpYr*np.sqrt(self.kb*T_init_grid/(self.mu_disc*self.mh))
    
                #Keplerian frequency
                omega_init_grid = np.sqrt(self.G*self.M_star/r_vals**3)
                
                #Gas scale height
                H_init_grid = c_s_init_grid/omega_init_grid
                
                #Turbulent viscosity
                nu_init_grid = self.alpha*c_s_init_grid*H_init_grid
                u_r_init = -3/2*nu_init_grid/r_vals            
                sigma_g_init = -self.M_0dot/(2*np.pi*r_vals*u_r_init)
                Z_disc_init_grid = self.ab_class.get_solid_frac(T_init_grid)
                
                dust_mass = integrate.simps(Z_disc_init_grid*sigma_g_init*2*np.pi*r_vals,r_vals,axis=0)
                
                M_dust_init[i_grid] = dust_mass
            
            M_dust_i = np.copy(M_dust_init)
            #St_vals = np.zeros(len(self.times))
            #M_pdot_vals = np.zeros(len(self.times))
        
        for i,t in enumerate(self.times):
            
            percent_done = np.around(i/len(self.times)*100,3)
                        
            print("\r{}".format(percent_done),end = "")
            
            r = y[:,:,1]
            M = y[:,:,0]
            
            r[r<0.1] = 0.1
            
            #Luminosity from Baraffe et al. (2015)
            lum_val = self.lums[i,:]
            #lum_val = 1            
            #Gas and pebble flux only depends on time so need to copy it to have same shape
            #as r and M (N_bodies,N_star)
            
            #Hartmann disc model. Calculate the viscosity at R1
            #and then the characteristic timescale ts
            #Temperature at disc edge will be that of irradiation temperature
            #This is lucky as otherwise we would get recursion error
            
            T1 = 150*(self.M_star/self.MSun2ME)**(-1/7)*self.lums[0,:]**(2/7)*(self.R1)**(-self.zeta)
            
            c_s1 = self.Mps2AUpYr*np.sqrt(self.kb*T1/(self.mu_disc*self.mh))
            omega1 = np.sqrt(self.G*self.M_star/self.R1**3)
            nu1 = self.alpha*c_s1**2/omega1
            
            ts = 1/(3*(2-self.gamma)**2)*self.R1**2/nu1
            
            exp = -(5/2-self.gamma)/(2-self.gamma)
            
            M_dotg = self.M_0dot*(t/ts+1)**exp
            
            M_dotg_ext = np.repeat(M_dotg,self.N_bodies,axis=0)
            
            M_dot = np.zeros((self.N_bodies,self.N_star))
            
            #Temperature from Ida et al. (2016)
            if self.viscous_heating:
                
                T_irr = 150*(self.M_star/self.MSun2ME)**(-1/7)*\
                    lum_val**(2/7)*\
                    (r)**(-self.zeta)
                    
                T_visc = 200*(self.M_star/self.MSun2ME)**(3/10)*\
                    (self.alpha/10**-3)**(-1/5)*\
                    (M_dotg_ext/self.MSun2ME/10**-8)**(2/5)*\
                    r**(-9/10)
                    
                T = np.max(np.stack((T_irr,T_visc)),axis=0)
                
            else:
                
                T = 150*(self.M_star/self.MSun2ME)**(-1/7)*\
                    lum_val**(2/7)*\
                    (r)**(-self.zeta)

            #Mean molecular weight of gas
            #Technically different at different locations in the disc
            #but does not vary significantly (Difference of ~few percent)
            #mu_disc = self.ab_class.get_mmw(T)
            
            #Dust to gas ratio in the disc
            Z = self.ab_class.get_solid_frac(T)
            #Z = self.Z
            
            #Sound speed
            c_s = self.Mps2AUpYr*np.sqrt(self.kb*T/(self.mu_disc*self.mh))

            #Keplerian frequency
            omega = np.sqrt(self.G*self.M_star/r**3)
            
            #Gas scale height
            H = c_s/omega
            
            #Hill radius
            R_H = (M/(3*self.M_star))**(1/3)*r
            
            #Turbulent viscosity
            nu = self.alpha*c_s*H
            
            #Gas accretion speed
            u_r = -3/2*nu/r
            
            #Keplerian velocity
            v_k = np.sqrt(self.G*self.M_star/r)
            
            #Column density of gas
            sigma_g = -M_dotg_ext/(2*np.pi*r*u_r)  
            
            #For constant flux ratio            
            if self.constant_flux:
                
                M_dotp = Z*M_dotg_ext

                St = self.St
            
            else:
                
                M_dotg_grid = np.repeat(M_dotg,N_grid-1,axis=0)
                
                if self.viscous_heating:
                    
                    T_irr_grid = 150*(self.M_star/self.MSun2ME)**(-1/7)*\
                        lum_val**(2/7)*\
                        (grid_points)**(-self.zeta)
                        
                    T_visc_grid = 200*(self.M_star/self.MSun2ME)**(3/10)*\
                        (self.alpha/10**-3)**(-1/5)*\
                        (M_dotg_grid/10**-8)**(2/5)*\
                        grid_points**(-9/10)
                    
                    T_grid = np.max(np.stack((T_irr_grid,T_visc_grid)),axis=0)
                    
                else:        
                    
                    T_grid = 150*(self.M_star/self.MSun2ME)**(-1/7)*lum_val**(2/7)*(grid_points)**(-self.zeta)
                
                Z_grid = self.ab_class.get_solid_frac(T_grid)
                
                #Z_grid = np.repeat(self.Z[np.newaxis,:],N_grid-1,axis=0)
                
                omega_grid = np.sqrt(self.G*self.M_star/grid_points**3)
                c_s_grid = self.Mps2AUpYr*np.sqrt(self.kb*T_grid/(self.mu_disc*self.mh))
                H_grid = c_s_grid/omega_grid
                
                v_k_grid = np.sqrt(self.G*self.M_star/grid_points)
                
                nu_grid = self.alpha*c_s_grid*H_grid
                u_r_grid = -3/2*nu_grid/grid_points
                
                sigma_g_grid = -M_dotg_grid/(2*np.pi*grid_points*u_r_grid)
                sigma_dust_grid = Z_grid*sigma_g_grid*M_dust_i/M_dust_init
                #rho_grid = sigma_g_grid/(np.sqrt(2*np.pi)*H_grid)
                
                #gp_grid = rho_grid*c_s_grid**2
                
                #gp_grid = np.array([np.interp(r_grid_gp[:,i],grid_points[:,i],gp_grid[:,i]) for i in range(self.N_star)]).T
                
                #eta_grid = (gp_grid[1:]-gp_grid[:-1])/dr/(2*rho_grid*omega_grid**2*grid_points)
                
                #vn_grid = (abs(eta_grid)*omega_grid*grid_points)
                vn_grid = 1/2*H_grid/grid_points*self.chi_val*c_s_grid
                
                t_growth = sigma_g_grid/sigma_dust_grid*1/omega_grid*(self.alpha_v/10**-4)**(-1/3)*(grid_points)**(1/3)
                f_dg = 30
                
                #Assume material density to be the mean of all our species
                #Initial stokes number corresponds to inital sizes (1micron)
                
                St0 = np.sqrt(2*np.pi)*6.685*10**(-18)*5.61*10**(11)*2.6/sigma_g_grid
                v_f = self.v_f*self.Mps2AUpYr

                Stini = St0*np.exp(t/t_growth)
                Stf = (v_f**2)/(3*self.alpha_v*c_s_grid**2)
                Stdf = v_f/(2*vn_grid)
                Stdrift = 1/(f_dg*vn_grid/v_k_grid)*sigma_dust_grid/sigma_g_grid
                St = np.min(np.stack((Stini,Stf,Stdf,Stdrift)),axis = 0)
                St[St<St0] = St0[St<St0]
                
                v_r0 = 2*vn_grid*St/(1+St**2)+abs(u_r_grid/(1+St**2))

                v_r_grid = np.min(np.stack((v_r0,grid_points/(f_dg*t_growth))),axis=0)
                M_pdoti = 2*np.pi*grid_points*v_r_grid*sigma_dust_grid
                
                #M_pdot_vals[i] = M_pdoti[(grid_points < 10.5) & (grid_points > 9.5)]
                #St_vals[i] = St[(grid_points < 10.5) & (grid_points > 9.5)]
                
                M_dust_i -= M_pdoti*dt
                
                counts = np.array([np.histogram(r[:,i],bins=r_grid[:,i])[0] for i in range(self.N_star)]).T

                M_dotp = np.array([np.repeat(M_pdoti[:,i],counts[:,i]) for i in range(self.N_star)]).T
                
                St = np.array([np.repeat(St[:,i],counts[:,i]) for i in range(self.N_star)]).T                
                
            #Pebble scale height
            H_p = H*np.sqrt(self.delta/(self.delta+St))
            #print(M_dotp)
            #Accretion radius
            R_acc = (St/0.1)**(1/3)*R_H
            
            if self.constant_flux:
                
                #Sub keplerian velocity            
                delta_v = 1/2*H/r*self.chi_val*c_s    
                
                #Radial drift of particles
                v_r = -2*delta_v/(self.St+self.St**-1)+u_r/(1+self.St**2)
                
                sigma_p = -M_dotp/(2*np.pi*r*v_r)
                
            else:
                
                #Column density of pebbles
                sigma_p = np.sqrt(2*M_dotp*sigma_g/(np.sqrt(3)*np.pi*self.eps_p*r*v_k))

            #Gas denisty in the midplane, from ideal gas law
            rho_midplane = sigma_g/(np.sqrt(2*np.pi)*H)
            
            #Pressure in midplane
            P = rho_midplane*self.kb*7.44*10**(-33)*T/(self.mu_disc*self.mh*1.674*10**-25)
            
            #Pebble isolation mass and gap mass
            #Isolation mass can be either from Bitsch or from gap mass
            #Gap mass can be from isolation mass (if it is from bitsch) or 
            #from kanagawa
            if self.iso == "bitsch":
            
                M_iso = 25*((H/r)/0.05)**3*\
                        (0.34*(np.log10(self.alpha_3)/np.log10(self.alpha_v))**4+0.66)*\
                        (1-((-self.chi_val+2.5)/6))
                M_gap = 2.3*M_iso
                
            else:
                
                M_gap = np.sqrt(1/0.04*(H/r)**5*self.alpha_v)*self.M_star
                M_iso = M_gap/2.3
                
            #Stratification integral - see Johansen et al (2015)
            #sint = H_p/(np.sqrt(2)*0.79*R_acc)*np.sqrt(np.pi)/2*\
            #       (erf(0.79*R_acc/(np.sqrt(2)*H_p))-erf(-0.79*R_acc/(np.sqrt(2)*H_p)))
            #Pebble accretion - see Johansen et al (2015)
            #M_dot3D = sint*np.pi*R_acc**2*sigma_p/(np.sqrt(2*np.pi)*H_p)*omega*R_acc

            M_dot_2D_filt = H_p < 2*np.sqrt(2*np.pi)/np.pi*R_acc

            M_dot[M_dot_2D_filt] = (2*R_acc*sigma_p*R_acc*omega)[M_dot_2D_filt]
            M_dot[~M_dot_2D_filt] = (np.pi*R_acc**2*sigma_p/(np.sqrt(2*np.pi)*H_p)*omega*R_acc)[~M_dot_2D_filt]

            #Calculate gas accretion for those which have reached pebble isolation mass
            iso_filt = M >= M_iso
            
            #pebble_filt_full = np.repeat(pebble_filt[np.newaxis,:],self.N_bodies,axis=0)
            
            gas_filt = iso_filt
            
            if np.any(gas_filt):
                
                #Array for storage of information
                #M_dot = np.zeros((self.N_bodies,self.N_star))
                
                #Kelvin-Heimholtz contraction
                dmdt_KH = 10**(-5)*(M/10)**(4)*(self.kappa/0.1)**(-1)
                
                #Disc accretion from infall into Hill sphere
                dmdt_disc = (H/r)**(-4)*(M/self.M_star)**(4/3)*M_dotg_ext/self.alpha*\
                            1/(1+(M/M_gap)**2)*0.29/(3*np.pi)
                            
                #Take the minimum of all three
                combined = np.array([M_dotg_ext,dmdt_KH,dmdt_disc])
                
                min_val = np.min(combined,axis = 0)

                #Masks those planets who have reached isolation mass and calculates
                #correct value
                
                M_dot[gas_filt] = min_val[gas_filt]
                
            #Migration                
            rdot_type1 = -self.kmig*M/self.M_star*(sigma_g*r**2/self.M_star)*(H/r)**(-2)*v_k
            r_dot = rdot_type1/(1+(M/(M_gap))**2)
            
            #Find those who have reached inner edge and stop the growth/migration
            r_in_filt = r <= self.r_in
            r_dot[r_in_filt] = 0
            M_dot[r_in_filt] = 0

            #Some planets should start growing at specific times set by
            #the variable starttimes. Whenever the current time is less than
            #the designated starttime - just set the derivative to 0
            t_filt = t<self.starttimes

            M_dot[t_filt] = 0
            r_dot[t_filt] = 0
            
            #Iterate over maximum disc lifetime and just set everything to 0 
            #for discs who are done
            gas_lifetime_filt = t > self.disc_lifetimes
            
            M_dot[:,gas_lifetime_filt] = 0
            r_dot[:,gas_lifetime_filt] = 0
            
            #Derivatives in mass and position
            ders = np.array([M_dot,r_dot]).swapaxes(0,1).swapaxes(1,2)

            #Actual step, use euler step as it is simple enough
            step = dt*ders
            
            #Mass accreted during timestep
            M_acc = step[:,:,0]
            
            if self.evap:
            
                #Bulk density, found from total mass and mass of solids
                #Volume of each species
                v_i = y_solid/self.density
                            
                #Volume fraction of each species
                vfrac = v_i/np.sum(v_i,axis = 2,keepdims=True)
                
                rho_bulk = np.sum(self.density*vfrac,axis=2)*5.61*10**8 #M_E/AU^3
                
                #Size of planet
                R = (3*M/(4*np.pi*rho_bulk))**(1/3)
                #Luminosity of planet from accretion
                L_pl = self.G*M_dot*M/R
                
                #Used to calculate pressure and thus temperature of radiative convective boundary
                #These will have slightly higher temperatures and so all solids won't be accreted
                nabla_0 = 3*self.kappa*P*L_pl/(64*np.pi*self.G*M*self.sigma_sb*T**4)
                nabla_inf = 1/(4-self.beta_opacity)
                nabla_ad = (self.gamma_ad-1)/self.gamma_ad
                P_rcb = P*(nabla_ad/nabla_0-nabla_ad/nabla_inf)/(1-nabla_ad/nabla_inf)
                
                T_rcb = T*(nabla_0/nabla_inf*(P_rcb/P-1)+1)**nabla_inf
                
                #If a planet hasn't started growing yet, nabla_0 = 0 and we divide by 0                
                #Therefore we set T_rcb to the disc temperature, could also set it to 0 as nothing would get accreted
                
                T_rcb[np.isnan(T_rcb)] = T[np.isnan(T_rcb)]
                
                #Calculate the solids accreted
                #Use T for gas as T_rcb is not valid with no pebble accretion
                abun_step_gas = self.ab_class.M_species(M_acc,T,gas=True)
                abun_step_solid = self.ab_class.M_species(M_acc,T_rcb,gas=False)
                
                #Need to reset the "step" i.e. accreted mass to the sum of solid after taking into account T_rcb
                step[:,:,0][~iso_filt] = np.sum(abun_step_solid,axis=2)[~iso_filt]
            
            else:
                
                abun_step_solid,abun_step_gas = self.ab_class.M_species(M_acc,T)
            
            #Add solid to those below isolation mass
            y_solid[~iso_filt] += abun_step_solid[~iso_filt]
            
            #Add gas to those above
            y_gas[iso_filt] += abun_step_gas[iso_filt]
                           
            y+= step
            
            if i != self.N_step-1:
                
                dt = self.times[i+1]-self.times[i]
            
            if self.store_full:
                
                self.data_out[i+1] = y
                self.solids[i+1] = y_solid
                self.gases[i+1] = y_gas
        
        if not self.store_full:
            
            self.data_out = y
            self.gases = y_gas
            self.solids = y_solid
        
        #plt.figure()
        #plt.plot(self.times,St_vals)
        #plt.xscale('log')
        #plt.yscale('log')

        #plt.figure()
        #plt.plot(self.times,M_pdot_vals)
        #plt.xscale('log')
        #plt.yscale('log')
        
        return self.times,self.data_out,self.gases,self.solids,self.ab_class.spec_names
        
        
class IntegrateFast:
    
    def __init__(self,theta,data,dt,init_abun,
                 starttimes,t0=0,t_end_full=3*10**6,
                 iso="bitsch",free_iron = False,carbon_grain=True,
                 store_full = False,evap=True,constant_flux=False,viscous_heating=False):
        '''
        Class for simulating growth of planets through pebble accretion
        Takes into account abundance and masses of stars and outputs
        compositions and final masses as well as positions of planets

        Parameters
        ----------
        theta : array
            Input disc/model parameters.
        data : array
            Initial positions and masses of the embryos. (N_bodies, N_star)
        dt : float
            timestep.
        init_abun : array
            initial abundances of all stars.
        t0 : float
            starting point of the integration.
        t_end : float
            end of integration.
        starttimes : array
            Sets the time when the embryos should start growing.
        iso : str, optional
            Set how I should calcualte the isolation mass.
            Either from Berts paper or from gap mass found by kanagawa.
            The default is "bitsch".
        free_iron : bool, optional
            Sets if I want free iron of fayalite and magnetite.
            The default is False.
        store_full : bool, optional
            Sets if I want to store all timesteps. The default is False.
        evap : bool, optional
            Sets if I consider evaporation of species during pebble accretion
        Raises
        ------
        ValueError
            If wrong iso type is set.

        Returns
        -------
        None.

        '''
        self.St,\
        self.Z,\
        self.alpha,\
        self.M_star,\
        self.zeta,\
        self.beta,\
        self.kappa,\
        self.delta,\
        self.v_f = theta
        
        self.data = data
        self.N_star = theta.shape[1]
        self.store_full = store_full
        self.starttimes = starttimes
                
        self.iso = iso
        
        self.evap = evap
        
        self.constant_flux = constant_flux
        
        self.viscous_heating = viscous_heating
        
        iso_names = ["bitsch","gap"]
        
        if iso not in iso_names:
            
            raise ValueError("Wrong iso name")
        
        #self.St = np.repeat(0.01,self.N_star)
        
        self.Mps2AUpYr = 2.108*10**-4
        self.MSun2ME = 1.9891*10**30/(5.9722*10**24)
        self.G = 1.184*10**-4 #AU^3 M_E^-1 yr^-2
        self.u = 1.66*10**-27 # kg
        self.kg2ME = 1.674*10**-25 
        
        self.alpha_3 = np.repeat(10**-3,self.N_star)
        self.alpha_v = np.repeat(1*10**-4,self.N_star)
        self.eps_d = 0.5/10
        
        self.ab_class = ab(init_abun,free_iron,carbon_grain)
        
        #Time stuff
        #Ended up not mattering too much but have logarithmic timesteps up until
        #a maximum
        
        self.disc_lifetimes = t_end_full
        self.t_end = np.max(t_end_full)
        self.max_dt=dt
        #times = np.logspace(0,np.log10(self.t_end),100)
        #times = times[np.insert(np.diff(times),0,0)<self.max_dt]
        #self.times = np.append(times,np.arange(np.max(times)+self.max_dt,self.t_end,self.max_dt))
        self.times = np.arange(t0,self.t_end,dt)
                
        self.N_step = len(self.times)

        self.M_gas = (0.1*(self.M_star/self.MSun2ME)**1.4)*self.MSun2ME
        self.lums = self.calc_lum(self.times)
        
        self.gamma=3/2-self.zeta
        
        #Set initial gas flux to be equal to 10 times the X-ray photoevaporation
        self.M_dotend = self.M_gphoto()
        self.M_0dot = 10*self.M_dotend
        
        #Inner edge of disc
        self.r_in = np.repeat(0.1,self.N_star) #AU
                    
        #Boltzmanns constant - used if I want to calculate pressures
        self.kb = 1.38*10**-23 #m^2 kg s^-2 K^-1
        self.sigma_sb = 2.978*10**-10
        self.mh = 1.00784*self.u #kg

        self.init_gas_mass = 0.1*(self.M_star/self.MSun2ME)**1.4
        
        #This can change - look into
        self.mu_disc = 2.3
               
        #Characteristic disc size
        self.R1 = self.get_R1()
        
        #Negative logarithmic pressure gradient
        self.chi_val = self.beta+self.zeta/2+3/2
        self.gamma = 3/2-self.zeta
        
        #Used to calculate the temperature at the RCB
        self.gamma_ad = 1.4
        self.beta_opacity = 2
        
        #Radial migration coefficient
        self.kmig = 2*(1.36+0.62*self.beta+0.43*self.zeta)
        
        #Abundance stuff
        #Need to calculate inital abundance of embryos, assume that their 
        #initial abundance reflect that of the solid material at the position
        #where they are intialised
        
        self.N_species = self.ab_class.N_species
        self.r_0 = data[:,:,1]
        self.M_0 = data[:,:,0]
        self.L_0 = self.calc_lum(self.starttimes)
        
        #If we have viscous heating, the temperature of the disc is equal to 
        #the maximum of viscous and irradtion heating
        
        #Otherwise only irradiation
        if self.viscous_heating:
            
            self.T_irr = 150*(self.M_star/self.MSun2ME)**(-1/7)*\
                self.L_0**(2/7)*\
                (self.r_0)**(-self.zeta)
                
            self.T_visc = 200*(self.M_star/self.MSun2ME)**(3/10)*\
                (self.alpha/10**-3)**(-1/5)*\
                (self.M_gdot(self.starttimes)/self.MSun2ME/10**-8)**(2/5)*\
                self.r_0**(-9/10)
            
            self.T_0 = np.max(np.stack((self.T_irr,self.T_visc)),axis=0)
            
        else:        
            
            self.T_0 = 150*(self.M_star/self.MSun2ME)**(-1/7)*self.L_0**(2/7)*(self.r_0)**(-self.zeta)
        
        self.N_bodies = self.r_0.shape[0]
        
        #Data of all chemical species used
        self.spec_data = pd.read_csv("species_data.csv",delimiter=";")
        
        #Species-specific constants needed for calculations
        #Need to extend them so they cover all bodies and all stars
        self.density = self.spec_data["density"].values
        self.density = np.repeat(self.density[np.newaxis,:],self.N_bodies,axis=0)
        self.density = np.repeat(self.density[:,np.newaxis,:],self.N_star,axis=1)

        #Initialise the solid abundances of the protoplanets
        #They do not have a gaseous component yet - set to zero
        self.init_solid = self.ab_class.M_species(self.M_0,self.T_0,gas=False)
        self.init_gas = np.zeros((self.N_bodies,self.N_star,self.N_species))
        
        if store_full:
            
            #Arrays for storage
            self.data_out = np.zeros((self.N_step+1,self.N_bodies,self.N_star,2))
            self.solids = np.zeros((self.N_step+1,self.N_bodies,self.N_star,self.N_species))
            self.gases = np.zeros((self.N_step+1,self.N_bodies,self.N_star,self.N_species))
        
    def M_gdot(self,t):
            
        gamma=3/2-self.zeta
        
        T = 150*(self.M_star/self.MSun2ME)**(-1/7)*self.calc_lum(0)**(2/7)*(self.R1)**(-self.zeta)
        c_s = self.Mps2AUpYr*np.sqrt(self.kb*T/(self.mu_disc*self.mh))
        omega = np.sqrt(self.G*self.M_star/self.R1**3)
        H = c_s/omega
        nu = self.alpha*c_s*H
        
        ts = 1/(3*(2-gamma)**2)*self.R1**2/nu
        
        exp = -(5/2-gamma)/(2-gamma)
        
        return self.M_0dot*(t/ts+1)**exp
    
    def calc_lum(self,times):
        '''
        
        Parameters
        ----------
        times : array
            array with the different timesteps .

        Returns
        -------
        lum_data : array
            Luminosities for each stars and each timestep.

        '''
        
        #Read isochrones and find each stellar mass value
        iso = pd.read_csv("mass_lum_isochrones.csv")
        isomass = iso["mass"].values        
        unique_isomass = np.unique(isomass)
        
        N_time = times.shape[0]
        
        lum_data = np.zeros((N_time,self.N_star))
        
        #Loop over each star
        for i,star in enumerate(self.M_star/self.MSun2ME):
            
            #Check for time array dimension
            if times.ndim==2:
                times_calc = times[:,i]
            else:
                times_calc = times
            
            #Find the closest mass value in isochrone and get L(t)
            diff = abs(unique_isomass-star)
            
            nearest = np.where(diff == np.min(diff))
            
            near_mass = unique_isomass[nearest]
            
            filt = isomass == near_mass
            
            iso_t = 10**iso["logt"].values[filt]
            iso_lum = 10**iso["lum"].values[filt]
            
            #Linearly interpolate - set bounds error to False to avoid nan
            #at edges - instead extrapolates
            #interp1d creates a function which can be called to get the values
            f = interpolate.interp1d(iso_t,iso_lum,bounds_error=False)
                       
            lum_data[:,i] = f(times_calc)
            
            lum_data[np.isnan(lum_data)] = f(min(iso_t))
        
        return lum_data
           
    def M_gphoto(self):
        '''
        Mass loss from X-rays as foundd by Owen et al. (2012)
        '''
        L_x = 10**(30.37+1.44*np.log10(self.M_star/self.MSun2ME))
        
        M_g = (6.25*10**(-9)*(self.M_star/self.MSun2ME)**(-0.068)*(L_x/10**30)**(1.14)).astype(float)
        
        return M_g*self.MSun2ME

    def get_R1(self):
        '''
        Calculate disc size from integrating the surface density and solving
        the analytical integral
        '''
    
        A = 39*self.kb*7.44*10**(-33)*self.alpha*150*self.calc_lum(np.array([0]))**(2/7)*(self.M_star/self.MSun2ME)**(-1/7)*self.M_gas
        
        B = 28*self.M_0dot*np.sqrt(self.G*self.M_star)*self.mu_disc*self.mh*self.kg2ME

        R_1 = (self.r_in**(13/14)+A/B)**(14/13)
        
        return R_1
    
    def integrate(self):

        y = self.data
        y_solid = self.init_solid
        y_gas = self.init_gas
        
        #If we have viscous heating, the temperature at the inner region of the disc,
        #might be high enough for all refractories to be evaporated
        #This means we cannot set the starting point of the pebble formation front
        #at a fixed value. Instead we set it to the maximum of inner edge and 
        #where the first species have condensed
        
        if self.viscous_heating:
            
            r_con = (1439/200*(self.M_star/self.MSun2ME)**(-3/10)*\
                    (self.alpha/10**-3)**(1/5)*\
                    (10*self.M_gphoto()/self.MSun2ME/10**-8)**(-2/5))**(-10/9)
            
            r_pf = np.max(np.stack((r_con,self.r_in)))
            
        else:
            
            r_pf = np.copy(self.r_in)
        
        if self.store_full:
            
            #data_out consists of (time,body,star,data), where the data is mass and radius
            self.data_out[0] = y
            self.solids[0] = y_solid
            self.gases[0] = y_gas
        
        dt = self.times[1]-self.times[0]
        
        for i,t in enumerate(self.times):
            
            percent_done = np.around(i/len(self.times)*100,3)
                        
            print("\r{}".format(percent_done),end = "")
            
            r = y[:,:,1]
            M = y[:,:,0]

            #Luminosity from Baraffe et al. (2015)
            lum_val = self.lums[i,:]
            
            #Gas and pebble flux only depends on time so need to copy it to have same shape
            #as r and M (N_bodies,N_star)
            
            #Hartmann disc model. Calculate the viscosity at R1
            #and then the characteristic timescale ts
            #Temperature at disc edge will be that of irradiation temperature
            #This is lucky as otherwise we would get recursion error
            
            T1 = 150*(self.M_star/self.MSun2ME)**(-1/7)*self.lums[0,:]**(2/7)*(self.R1)**(-self.zeta)
            c_s1 = self.Mps2AUpYr*np.sqrt(self.kb*T1/(self.mu_disc*self.mh))
            omega1 = np.sqrt(self.G*self.M_star/self.R1**3)
            nu1 = self.alpha*c_s1**2/omega1
            
            ts = 1/(3*(2-self.gamma)**2)*self.R1**2/nu1
            
            exp = -(5/2-self.gamma)/(2-self.gamma)
            
            M_dotg = self.M_0dot*(t/ts+1)**exp
            
            M_dotg = np.repeat(M_dotg,self.N_bodies,axis=0)
            
            M_dot = np.zeros((self.N_bodies,self.N_star))
            
            #Temperature from Ida et al. (2016)
            if self.viscous_heating:
                
                T_irr = 150*(self.M_star/self.MSun2ME)**(-1/7)*\
                    lum_val**(2/7)*\
                    (r)**(-self.zeta)
                    
                T_visc = 200*(self.M_star/self.MSun2ME)**(3/10)*\
                    (self.alpha/10**-3)**(-1/5)*\
                    (M_dotg/self.MSun2ME/10**-8)**(2/5)*\
                    r**(-9/10)
                    
                T = np.max(np.stack((T_irr,T_visc)),axis=0)
                
            else:
                
                T = 150*(self.M_star/self.MSun2ME)**(-1/7)*\
                    lum_val**(2/7)*\
                    (r)**(-self.zeta)

            #Mean molecular weight of gas
            #Technically different at different locations in the disc
            #but does not vary significantly (Difference of ~few percent)
            #mu_disc = self.ab_class.get_mmw(T)
            
            #Dust to gas ratio in the disc
            Z = self.ab_class.get_solid_frac(T)
            #Z = self.Z
            
            #Sound speed
            c_s = self.Mps2AUpYr*np.sqrt(self.kb*T/(self.mu_disc*self.mh))
            
            #Keplerian frequency
            omega = np.sqrt(self.G*self.M_star/r**3)
            
            #Gas scale height
            H = c_s/omega
            
            #Pebble scale height
            H_p = H*np.sqrt(self.delta/(self.delta+self.St))
            
            #Hill radius
            R_H = (M/(3*self.M_star))**(1/3)*r
            
            #Accretion radius
            R_acc = (self.St/0.1)**(1/3)*R_H
            
            #Turbulent viscosity
            nu = self.alpha*c_s*H
            
            #Gas accretion speed
            u_r = -3/2*nu/r
            
            #Sub keplerian velocity            
            delta_v = 1/2*H/r*self.chi_val*c_s    
            
            #Radial drift of particles
            v_r = -2*delta_v/(self.St+self.St**-1)+u_r/(1+self.St**2)
            
            #Keplerian velocity
            v_k = np.sqrt(self.G*self.M_star/r)
            
            #For constant flux ratio            
            if self.constant_flux:
                
                M_dotp = Z*M_dotg

            else:
                            
                #Pebble flux from Lambrechts & Johansen (2014)
                #Idea is that we calculate the time it takes for dust to grow
                #to pebbles which is different at different r. This means that a
                #pebble formation front is moving outwards in the disc
                #the position of the front (r_pf) is not only dependent on time
                #but also on the metallicity which means that I cannot calculate
                #it analytically if I want to include evaporation of species
                #as Z=Z(r). 
                
                #Start by calculating T and Z at current r_pf
                if self.viscous_heating:

                    T_irr_rpf = 150*(self.M_star/self.MSun2ME)**(-1/7)*\
                        lum_val**(2/7)*\
                        (r_pf)**(-self.zeta)
                        
                    T_visc_rpf = 200*(self.M_star/self.MSun2ME)**(3/10)*\
                        (self.alpha/10**-3)**(-1/5)*\
                        (M_dotg[0]/self.MSun2ME/10**-8)**(2/5)*\
                        r_pf**(-9/10)

                    T_rpf = np.max(np.stack((T_irr_rpf,T_visc_rpf)),axis=0)

                else:
                    
                    T_rpf = 150*(self.M_star/self.MSun2ME)**(-1/7)*\
                        lum_val**(2/7)*\
                        (r_pf)**(-self.zeta)

                Z_rpf = self.ab_class.get_solid_frac(T_rpf.reshape(1,self.N_star))[0]

                #The derivative of r_pf
                dr_pfdt = (2/3)*(3/16)**(1/3)*(self.G*self.M_star)**(1/3)*\
                          (self.eps_d*Z_rpf)**(2/3)*t**(-1/3) 
                
                #If first time step: set to 0, otherwise it will be division by zero
                if t == 0:
                    
                    dr_pfdt = np.zeros(self.N_star)
                
                #Extend to cover all bodies
                r_pf_ext = np.repeat(r_pf[np.newaxis,:],self.N_bodies,axis=0)
                dr_pfdt_ext = np.repeat(dr_pfdt[np.newaxis,:],self.N_bodies,axis=0)
                T_rpf_ext = np.repeat(T_rpf[np.newaxis,:],self.N_bodies,axis=0)
                
                #Calculate the rest of quantities for the pebble flux (sigma_g at r_pf)
                cs_rpf=self.Mps2AUpYr*np.sqrt(self.kb*T_rpf_ext/(self.mu_disc*self.mh))
    
                omega_rpf = np.sqrt(self.G*self.M_star/r_pf_ext**3)
                H_rpf = cs_rpf/omega_rpf
                u_rpf = 3/2*self.alpha*cs_rpf*H_rpf/r_pf_ext
                
                M_dotp = Z*dr_pfdt_ext*M_dotg/u_rpf
    
                if t == 0:
                    
                    M_dotp = np.zeros((self.N_bodies,self.N_star))
    
                #Stops if r_g>R1
                #Planetesimals outside of pebble formation front can't accrete pebbles
                #As no pebbles are dirfting inwards yet
                M_dotp[r>r_pf_ext]=0
                pebble_filt = r_pf>self.R1[0]
    
                M_dotp[:,pebble_filt]*=np.exp(-(r_pf-self.R1[0])/self.R1[0])[pebble_filt]

            #Column density of gas
            sigma_g = -M_dotg/(2*np.pi*r*u_r)  
   
            #Column density of pebbles
            sigma_p = -M_dotp/(2*np.pi*r*v_r)

            if not self.constant_flux:
                
                #Advance the pebble formation front
                r_pf+=dr_pfdt*dt
                        
            #Gas denisty in the midplane, from ideal gas law
            rho_midplane = sigma_g/(np.sqrt(2*np.pi)*H)
            #Pressure in midplane
            P = rho_midplane*self.kb*7.44*10**(-33)*T/(self.mu_disc*self.mh*1.674*10**-25)
            
            #Pebble isolation mass and gap mass
            #Isolation mass can be either from Bitsch or from gap mass
            #Gap mass can be from isolation mass (if it is from bitsch) or 
            #from kanagawa
            if self.iso == "bitsch":
            
                M_iso = 25*((H/r)/0.05)**3*\
                        (0.34*(np.log10(self.alpha_3)/np.log10(self.alpha_v))**4+0.66)*\
                        (1-((-self.chi_val+2.5)/6))
                M_gap = 2.3*M_iso
                
            else:
                
                M_gap = np.sqrt(1/0.04*(H/r)**5*self.alpha_v)*self.M_star
                M_iso = M_gap/2.3
                
            #Stratification integral - see Johansen et al (2015)
            #sint = H_p/(np.sqrt(2)*0.79*R_acc)*np.sqrt(np.pi)/2*\
            #       (erf(0.79*R_acc/(np.sqrt(2)*H_p))-erf(-0.79*R_acc/(np.sqrt(2)*H_p)))
            #Pebble accretion - see Johansen et al (2015)
            #M_dot3D = sint*np.pi*R_acc**2*sigma_p/(np.sqrt(2*np.pi)*H_p)*omega*R_acc

            M_dot_2D_filt = H_p < 2*np.sqrt(2*np.pi)/np.pi*R_acc

            M_dot[M_dot_2D_filt] = (2*R_acc*sigma_p*R_acc*omega)[M_dot_2D_filt]
            M_dot[~M_dot_2D_filt] = (np.pi*R_acc**2*sigma_p/(np.sqrt(2*np.pi)*H_p)*omega*R_acc)[~M_dot_2D_filt]

            #Calculate gas accretion for those which have reached pebble isolation mass
            iso_filt = M >= M_iso
            
            #pebble_filt_full = np.repeat(pebble_filt[np.newaxis,:],self.N_bodies,axis=0)
            
            gas_filt = iso_filt
            
            if np.any(gas_filt):
                
                #Array for storage of information
                #M_dot = np.zeros((self.N_bodies,self.N_star))
                
                #Kelvin-Heimholtz contraction
                dmdt_KH = 10**(-5)*(M/10)**(4)*(self.kappa/0.1)**(-1)
                
                #Disc accretion from infall into Hill sphere
                dmdt_disc = (H/r)**(-4)*(M/self.M_star)**(4/3)*M_dotg/self.alpha*\
                            1/(1+(M/M_gap)**2)*0.29/(3*np.pi)
                            
                #Take the minimum of all three
                combined = np.array([M_dotg,dmdt_KH,dmdt_disc])
                
                min_val = np.min(combined,axis = 0)

                #Masks those planets who have reached isolation mass and calculates
                #correct value
                
                M_dot[gas_filt] = min_val[gas_filt]
                
            #Migration                
            rdot_type1 = -self.kmig*M/self.M_star*(sigma_g*r**2/self.M_star)*(H/r)**(-2)*v_k
            r_dot = rdot_type1/(1+(M/(M_gap))**2)
            
            #Find those who have reached inner edge and stop the growth/migration
            r_in_filt = r < self.r_in
            r_dot[r_in_filt] = 0
            M_dot[r_in_filt] = 0

            #Some planets should start growing at specific times set by
            #the variable starttimes. Whenever the current time is less than
            #the designated starttime - just set the derivative to 0
            t_filt = t<self.starttimes

            M_dot[t_filt] = 0
            r_dot[t_filt] = 0
            
            #Iterate over maximum disc lifetime and just set everything to 0 
            #for discs who are done
            gas_lifetime_filt = t > self.disc_lifetimes
            
            M_dot[:,gas_lifetime_filt] = 0
            r_dot[:,gas_lifetime_filt] = 0
            
            #Derivatives in mass and position
            ders = np.array([M_dot,r_dot]).swapaxes(0,1).swapaxes(1,2)

            #Actual step, use euler step as it is simple enough
            step = dt*ders
            
            #Mass accreted during timestep
            M_acc = step[:,:,0]
            
            if self.evap:
            
                #Bulk density, found from total mass and mass of solids
                #Volume of each species
                v_i = y_solid/self.density
                            
                #Volume fraction of each species
                vfrac = v_i/np.sum(v_i,axis = 2,keepdims=True)
                
                rho_bulk = np.sum(self.density*vfrac,axis=2)*5.61*10**8 #M_E/AU^3
                
                #Size of planet
                R = (3*M/(4*np.pi*rho_bulk))**(1/3)
                #Luminosity of planet from accretion
                L_pl = self.G*M_dot*M/R
                
                #Used to calculate pressure and thus temperature of radiative convective boundary
                #These will have slightly higher temperatures and so all solids won't be accreted
                nabla_0 = 3*self.kappa*P*L_pl/(64*np.pi*self.G*M*self.sigma_sb*T**4)
                nabla_inf = 1/(4-self.beta_opacity)
                nabla_ad = (self.gamma_ad-1)/self.gamma_ad
                P_rcb = P*(nabla_ad/nabla_0-nabla_ad/nabla_inf)/(1-nabla_ad/nabla_inf)
                
                T_rcb = T*(nabla_0/nabla_inf*(P_rcb/P-1)+1)**nabla_inf
                
                #If a planet hasn't started growing yet, nabla_0 = 0 and we divide by 0                
                #Therefore we set T_rcb to the disc temperature, could also set it to 0 as nothing would get accreted
                
                T_rcb[np.isnan(T_rcb)] = T[np.isnan(T_rcb)]
                
                #Calculate the solids accreted
                #Use T for gas as T_rcb is not valid with no pebble accretion
                abun_step_gas = self.ab_class.M_species(M_acc,T,gas=True)
                abun_step_solid = self.ab_class.M_species(M_acc,T_rcb,gas=False)
                
                #Need to reset the "step" i.e. accreted mass to the sum of solid after taking into account T_rcb
                step[:,:,0][~iso_filt] = np.sum(abun_step_solid,axis=2)[~iso_filt]
            
            else:
                
                abun_step_solid,abun_step_gas = self.ab_class.M_species(M_acc,T)
            
            #Add solid to those below isolation mass
            y_solid[~iso_filt] += abun_step_solid[~iso_filt]
            
            #Add gas to those above
            y_gas[iso_filt] += abun_step_gas[iso_filt]
                           
            y+= step
            
            if i != self.N_step-1:
                
                dt = self.times[i+1]-self.times[i]
            
            if self.store_full:
                
                self.data_out[i+1] = y
                self.solids[i+1] = y_solid
                self.gases[i+1] = y_gas
        
        if not self.store_full:
            
            self.data_out = y
            self.gases = y_gas
            self.solids = y_solid
        
        return self.times,self.data_out,self.gases,self.solids,self.ab_class.spec_names
        
        