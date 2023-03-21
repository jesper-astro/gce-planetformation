# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 09:39:37 2022

@author: Jesper Nielsen
"""

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import warnings
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from chemistry import Abundance as ab
from scipy import interpolate
from scipy import integrate

def rescale(arr,newmin,newmax):
    
    max_arr = np.max(arr)
    min_arr = np.min(arr)

    return (newmax-newmin)/(max_arr-min_arr)*(arr-min_arr)+newmin

def draw_log(low=0.1,high=1,size = None):
    
    return np.random.uniform(np.log10(low),np.log10(high),size)

def get_trunc(shape,lower,upper,mean,scale = 1):
    '''
    Draw values from a truncated normal distribution
    '''

    try:
        
        dim = len(shape)
        
        if dim != 1:
            
            N_x = shape[0]
            N_y = shape[1]
            N_draw = N_x*N_y
        
        else:
            
            N_draw = shape[0]
            
    except TypeError:
        
        dim = 1
        N_draw = shape
    
    
    a,b = (lower - mean) / scale, (upper - mean) / scale
    
    if dim == 1:
        
        draw = truncnorm(a,b,loc = mean,scale = scale).rvs(N_draw)
    
    elif dim == 2:

        draw = np.zeros((N_x,N_y))
        
        for i in range(N_x):
                    
            draw[i] = truncnorm(a,b,loc = mean,scale = scale).rvs(N_y)
    
    else:
        
        raise ValueError("shape must be array of length 2 or int")
        
    return draw

def find_nearest(arr,val,return_value = False):
    
    '''
    Finds in array arr the nearest value to val and returns its index
    
    if return_value is true, then it instead returns the value in the array
    nearest to val
    
    '''
    
    nearest_value = arr[np.abs(arr-val).argmin()]
    
    if return_value:
        
        return nearest_value
    
    idx = np.where(arr==nearest_value)
    
    return idx[0][0]

def get_abun(val,solar_val):
    '''
    Return the absolute number fraction of element and hydrogen
    
    Inputs in typical notation and solar abundances in absolute fraction
    
    Solar abundances is from Asplund et al. (2009)
    '''
    return 10**(val+np.log10(solar_val))

def get_abun_solar(val,solar_val):
    
    '''
    Calculates [X/H] from the fraction X/H
    '''
    
    return np.log10(val)-np.log10(solar_val)

def get_ironratio(val,feh,solar_val,log_scale = False):
    '''
    Calculates [X/Fe] from Fe/H and X/H
    '''
    solar_abun = pd.read_csv("solar_abun.csv")
    feh_sun = solar_abun["Fe/H"].loc[0]

    if log_scale:

        val = 10**(val)*solar_val
        feh = 10**(feh)*feh_sun

    #(X/Fe)_sun
    val_fe_sun = solar_val/feh_sun
        
    val_out = np.log10(val/feh)-np.log10(val_fe_sun)
    
    return val_out

def xfe_to_xh(xfe,feh,solar_xh,solar_feh):
    '''
    Calculates [X/H] from [X/Fe]

    '''    
    n_fe_h = (10**feh)*solar_feh
    solar_x_fe = solar_xh/solar_feh
    
    n_x_fe = (10**xfe)*solar_x_fe
    
    return np.log10(n_x_fe*n_fe_h)-np.log10(solar_xh) 

def get_element_mass(final_gas_mass,final_solid_mass,names):
    '''
    Calculate the mass of elements based on gas masses of species and 
    solid masses of species
    Also needs species names as input in correct order
    '''
    final_gas_mass = final_gas_mass.swapaxes(0,1)
    final_solid_mass = final_solid_mass.swapaxes(0,1)
    
    N_pl = final_gas_mass.shape[1]
    N_star = final_gas_mass.shape[0]
    
    element_data = pd.read_csv("./element_data.csv",delimiter = ";")
    species_data = pd.read_csv("./species_data.csv",delimiter = ";")
    
    store_solid = np.zeros((len(element_data),N_star,N_pl))
    store_gas = np.zeros((len(element_data),N_star,N_pl))
    gas_mass = dict(zip(element_data["Element"].values,store_gas))
    solid_mass = dict(zip(element_data["Element"].values,store_solid))
    
    for i,name in enumerate(names):
        
        split_name = name.split("-")
        spec_mass = species_data["mass"].loc[species_data["species"] == name].values[0]

        for j,el_name in enumerate(split_name):
    
            if len(el_name) == 2:
                
                try:
                
                    el_nr = int(el_name[1])
                    el_filt = element_data["Element"] == el_name[0]
                    el_mass = el_nr*element_data["Mass"].loc[el_filt].values[0]
                    solid_mass[el_name[0]]+= el_mass/spec_mass*final_solid_mass[:,:,i]
                    gas_mass[el_name[0]]+= el_mass/spec_mass*final_gas_mass[:,:,i]
                    
                except ValueError:
    
                    el_filt = element_data["Element"] == el_name                
                    el_mass = element_data["Mass"].loc[el_filt].values[0]
                    solid_mass[el_name]+=el_mass/spec_mass*final_solid_mass[:,:,i]
                    gas_mass[el_name]+=el_mass/spec_mass*final_gas_mass[:,:,i]
                                       
            elif len(el_name) == 1:
                
                el_filt = element_data["Element"] == el_name
                el_mass = element_data["Mass"].loc[el_filt].values[0]
                solid_mass[el_name]+=el_mass/spec_mass*final_solid_mass[:,:,i]
                gas_mass[el_name]+=el_mass/spec_mass*final_gas_mass[:,:,i]
            
            elif len(el_name) == 3:

                el_filt = element_data["Element"] == el_name[:2]
                el_mass = int(el_name[2])*element_data["Mass"].loc[el_filt].values[0]
                solid_mass[el_name[:2]]+=el_mass/spec_mass*final_solid_mass[:,:,i]
                gas_mass[el_name[:2]]+=el_mass/spec_mass*final_gas_mass[:,:,i]
            
            else:
                
                print('Warning')
    
    return solid_mass,gas_mass

def lin_inpol(x,x_data,y_data):
    
    '''
    Linearly interpolate any data
    '''
    
    x0,x1 = x_data
    y0,y1 = y_data
    
    y = y0+(x-x0)*(y1-y0)/(x1-x0)
    
    return y

def HZ(Teff,L,M_pl):
    '''
    Calcualtes the inner and outer habitable zone limits based on stellar
    parameters and planet masses
    
    Polynomial coefficients from Kopparapu et al., (2014)
    '''
    
    N_star, N_pl = M_pl.shape
    
    T = Teff-5760
    
    T = T.reshape(N_star,1)
    L = L.reshape(N_star,1)
    
    M_lim1  = (M_pl >= 0.1) & (M_pl < 1)
    M_lim2  = (M_pl >=1) & (M_pl <= 5)

    a = np.zeros((N_star,N_pl))
    b = np.zeros((N_star,N_pl))
    S = np.zeros((N_star,N_pl))
    c = np.zeros((N_star,N_pl))
    d = np.zeros((N_star,N_pl))
    
    x_dat1 = (0.1,1)
    x_dat2 = (1,5)
    
    S_dat1 = (0.99,1.107)
    S_dat2 = (1.107,1.188)
    
    a_dat1 = (1.209*10**-4,1.332*10**-4)
    a_dat2 = (1.332*10**-4,1.433*10**-4)
    
    b_dat1 = (1.404*10**-8,1.58*10**-8)
    b_dat2 = (1.58*10**-8,1.707*10**-8)
    
    c_dat1 = (-7.418*10**-12,-8.308*10**-12)
    c_dat2 = (-8.308*10**-12,-8.968*10**-12)
  
    d_dat1 = (-1.713*10**-15,-1.931*10**-15)
    d_dat2 = (-1.931*10**-15,-2.084*10**-15)

    a[M_lim1] = lin_inpol(M_pl[M_lim1],x_dat1,a_dat1)
    a[M_lim2] = lin_inpol(M_pl[M_lim2],x_dat2,a_dat2)
    
    b[M_lim1] = lin_inpol(M_pl[M_lim1],x_dat1,b_dat1)
    b[M_lim2] = lin_inpol(M_pl[M_lim2],x_dat2,b_dat2)
    
    c[M_lim1] = lin_inpol(M_pl[M_lim1],x_dat1,c_dat1)
    c[M_lim2] = lin_inpol(M_pl[M_lim2],x_dat2,c_dat2)
    
    d[M_lim1] = lin_inpol(M_pl[M_lim1],x_dat1,d_dat1)
    d[M_lim2] = lin_inpol(M_pl[M_lim2],x_dat2,d_dat2)
    
    S[M_lim1] = lin_inpol(M_pl[M_lim1],x_dat1,S_dat1)
    S[M_lim2] = lin_inpol(M_pl[M_lim2],x_dat2,S_dat2)
    
    Seff_in = S+a*T+b*T**2+c*T**3+d*T**4
    
    S_out = 0.356
    a_out = 6.171*10**-5
    b_out = 1.698*10**-9
    c_out = -3.198*10**-12
    d_out = -5.575*10**-16
    
    Seff_out = S_out+a_out*T+b_out*T**2+c_out*T**3+d_out*T**4

    with warnings.catch_warnings():
        
        warnings.filterwarnings("ignore","divide by zero encountered in true_divide")        
        d_in = (L/Seff_in)**(0.5)
        d_out = (L/Seff_out)**(0.5)
    
    return d_in,d_out

def lum(M_star):
    '''
    MLR from Eker et al.,(2015)
    '''
    try:
        
        alpha = np.zeros(len(M_star))

    except TypeError:
        
        alpha = np.zeros(1)
    
    filt1 = (M_star <= 1.05) & (M_star > 0.38)
    filt2 = (M_star <= 2.4) & (M_star > 1.05)
    filt3 = (M_star <= 7) & (M_star > 2.4)
    filt4 = (M_star < 32) & (M_star > 7)
    
    alpha[filt1] = 4.841
    alpha[filt2] = 4.328
    alpha[filt3] = 3.962
    alpha[filt4] = 2.726
    
    if np.any(alpha == 0):
        
        raise ValueError("One star is not in the range 0.38-32 M_sun")
    
    return M_star**(alpha)

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

def lin_func(x,k,m):
    
    return k*x+m

def scale_data(feh,mgfe,new_feh,mgfe_err = None,
               mass = None,names = None,scatter = False,
               df = False):
    
    '''
    Outputs stars scaled from input feh and mgfe 
    Input is in log notation scaled with solar i.e. [Fe/H] and [Mg/Fe]
    
    Can take errors using mgfe_err
    
    mass and names can be inputted to create an array with complete data
    
    set df=True to create a pandas dataframe of the data, headers are 
    handled automatically
    
    scatter sets whether or not scatter is introduced to the new abundances
    Scatter is drawn from a normal distibution with mean 0 and width equal
    to the mean of the residuals from the fit
    '''
    
    solar_abun = pd.read_csv("solar_abun.csv")
    el_data = pd.read_csv("element_data.csv",delimiter = ";")
    
    try:
        
        N_star = len(new_feh)

    except TypeError:
        
        N_star = 1
        
    n_fe_h = 10**(feh)*solar_abun["Fe/H"].loc[0]

    n_x_fe_sun = solar_abun.loc[0]/solar_abun["Fe/H"].loc[0]

    n_mg_fe = 10**(mgfe)*n_x_fe_sun["Mg/H"]

    mgh = np.log10(n_mg_fe*n_fe_h)-np.log10(solar_abun["Mg/H"].loc[0])
    
    param,cov = curve_fit(lin_func,feh,mgh,sigma=mgfe_err)

    elements = ["C","N","Mg","Si","O","Fe","S","He"]
    
    mock_in = np.zeros((N_star,len(elements)))
    
    columns = []
    
    res = mgh-lin_func(feh,*param)
    
    for i_el,element in enumerate(elements):
        
        k = np.random.normal(param[0],cov[0,0],N_star)
        m = np.random.normal(param[1],cov[1,1],N_star)
        
        if scatter:
            
            scatter_vals = np.random.normal(0,np.mean(abs(res)),N_star)
        
        else:
            
            scatter_vals = 0
            
        new_data = lin_func(new_feh,k,m)+scatter_vals
        
        columns.append(element+"/H")
            
        if element == "He":
            
            continue
                
        elif element != "Fe":
        
            mock_in[:,i_el] = get_abun(new_data,solar_abun[element+'/H'].loc[0])
        
        else:
            
            mock_in[:,i_el] = get_abun(new_feh, solar_abun["Fe/H"].loc[0])
    
    Y = (-0.0564*new_feh+0.24)
    
    n_z_h = 0
    
    for i,el in enumerate(elements):
        if el == "He":
            continue
        idx = np.where(el_data["Element"]==el)[0][0]
        
        n_z_h+=el_data["Mass"].values[idx]*mock_in[:,i_el]
    
    n_h_tot = (1-Y)/(1+n_z_h)
    
    mock_in[:,-1] = Y/4/n_h_tot
    
    if df:
        
        if mass is not None:
            
            columns+=["mass"]
            
            mass_in = mass
            
            mock_in = np.append(mock_in,mass_in.reshape(N_star,1),axis = 1)
                    
        mock_df = pd.DataFrame(mock_in,columns = columns)
        
        if names is not None:
            
            mock_df["name"] = names

        return mock_df
    
    return mock_in

def rename_species(spec_names):
    
    new_spec_names = []
    for string in spec_names:
        
        new_string = ''
        
        for char in string:
            
            if char.isnumeric(): 
                
                new_string += '$_{}$'.format(char)
            
            else:
                
                if char=='-':
                    
                    continue
                
                new_string+= char
                
        new_spec_names.append(new_string)
    
    return new_spec_names


class Calculator:
    
    def __init__(self,theta,abun = None,viscous_heating=False):
        
        #Shape of arrays should be (N_points,N_star) where
        #N_points can be 
        self.St,\
        self.Z,\
        self.alpha,\
        self.M_star,\
        self.zeta,\
        self.beta,\
        self.kappa,\
        self.delta,\
        self.v_f = theta
        
        self.N_star = theta.shape[1]
                
        self.viscous_heating=viscous_heating
        
        self.Mps2AUpYr = 2.108*10**-4
        self.MSun2ME = 1.9891*10**30/(5.9722*10**24)
        self.G = 1.184*10**-4 #AU^3 M_E^-1 yr^-2
        self.u = 1.66*10**-27 # kg
        self.kg2ME = 1.674*10**-25 
        
        if abun is not None:
            
            if abun.shape[0] != self.N_star:
                
                raise ValueError('Abundance data and input parameters do not share the same number of stars!')
            
            self.ab_class = ab(abun)
            self.Z = self.ab_class.find_Z()
        
        self.spec_data = pd.read_csv("species_data.csv",delimiter = ";")
        self.N_species = len(self.spec_data)
        
        #Constants needed for calculations
        self.density = self.spec_data["density"].values
        
        self.alpha_3 = np.repeat(10**-3,self.N_star)
        self.alpha_v = np.repeat(1*10**-4,self.N_star)
        self.eps_d = 0.5/11.5
        
        gas_norm = 0.1
        
        if gas_norm != 0.1:
            
            print('Warning gas norm = {}'.format(gas_norm))
        #Starting disc accretion [M_E yr^-1]
        #self.M_0dot = np.ones(self.N_star)*10**-7*self.MSun2ME
        #Final disc accretion
        self.M_enddot = self.M_gphoto()*self.MSun2ME
        self.M_0dot = 10*self.M_enddot
        self.M_gas = gas_norm*(self.M_star/self.MSun2ME)**1.4*self.MSun2ME
        
        #Inner edge of disc
        self.r_in = np.repeat(0.1,self.N_star) #AU
        
        #Boltzmanns constant - used if I want to calculate pressures
        self.kb = 1.38*10**-23 #m^2 kg s^-2 K^-1
        self.sigma_sb = 2.978*10**-10
        
        self.mu_disc = 2.3
        
        self.mh = 1.00784*self.u #kg
        
        self.chi = self.beta+self.zeta/2+3/2
        self.gamma = 3/2-self.zeta
        
        self.gamma_ad = 1.4
        self.beta_opacity = 2
        self.R1 = self.get_R1()
        #Calculate the characteristic radius.
        #Found by forcing the gas accretion to
        #go from M_0dot to M_enddot from 0 to 3 Myr
                
        self.kmig = 2*(1.36+0.62*self.beta+0.43*self.zeta)
    
    def check_input_shape(self,arr,return_1D = False):
        
        if type(arr) is not int and type(arr) is not float and type(arr) is not np.float64:
            
            if arr.ndim == 1:
                
                arr = np.repeat(arr[:,np.newaxis],self.N_star,axis=1)
        
            if return_1D:
                
                return arr[:,0]
        
        return arr
    
    def r_pf_init(self):
        
        if self.viscous_heating:
        
            r_pf = (1300/200*(self.M_star/self.MSun2ME)**(-3/10)*\
                    (self.alpha/10**-3)**(1/5)*\
                    (10*self.M_gphoto()/10**-8)**(-2/5))**(-10/9)
        
        else:
            
            r_pf = np.copy(self.r_in)
        
        return r_pf.reshape(1,self.N_star)
           
    def M_gphoto(self):
        
        L_x = 10**(30.37+1.44*np.log10(self.M_star/self.MSun2ME))
        
        return (6.25*10**(-9)*(self.M_star/self.MSun2ME)**(-0.068)*((L_x/10**30))**(1.14)).astype(float)
    
    def v_k(self,r):
        '''
        Keplerian speed

        '''
        r = self.check_input_shape(r)
        
        return np.sqrt(self.G*self.M_star/r)
    
    def omega(self,r):
        '''
        Keplerian frequency

        Parameters
        ----------
        r: Radial location of planet

        '''
        
        r = self.check_input_shape(r)
        
        return np.sqrt(self.G*self.M_star/r**3)
    
    def R_H(self,M,r):
        '''
        Hill radius
        
        Parameters
        ----------
        M: Mass of the protoplanet
        M_star: Host star mass
        r: Radial location of planet
    
        '''
        
        M = self.check_input_shape(M)
        r = self.check_input_shape(r)
        
        return (M/(3*self.M_star))**(1/3)*r
    
    def R_acc(self,M,r):
        
        return (self.St/0.1)**(1/3)*self.R_H(M, r)    
    
    def c_s(self,r,t):
        '''
        Gas sound speed
        
        Parameters
        ----------
        r: Radial location (IN AU)

        '''
        
        return self.Mps2AUpYr*np.sqrt(self.kb*self.T(r,t)/(self.mu_disc*self.mh))
    
    def H(self,r,t):
        '''
        Gas scale height
        
        Parameters
        ----------
        r: Radial location
        '''
        
        return self.c_s(r,t)/self.omega(r)
    
    def H_p(self,r,t):
        
        return self.H(r,t)*np.sqrt(self.delta/(self.delta+self.St))
    
    def nu(self,r,t):
        '''
        Turbulent viscosity
        '''
        
        return self.alpha*self.c_s(r,t)*self.H(r,t)
    
    def u_r(self,r,t):
        '''
        Gas accretion speed

        Parameters
        ----------
        nu: Turbulent viscosity
        r: Radial location

        '''
        r = self.check_input_shape(r)
        return -3/2*self.nu(r,t)/r
    
    def delta_v(self,r,t):
        '''
        Sub-Keplerian speed
    
        Parameters
        ----------
        H: Gas scale height
        r: Radial location
        chi: Logarithmic pressure gradient in the midplane
        c_s : Gas sound speed
    
        '''
        r = self.check_input_shape(r)
        return 1/2*self.H(r,t)/r*self.chi*self.c_s(r,t)
    
    def v_r(self,r,t):
        '''
        Radial Drift of particles
        
        Parameters
        ----------
        delta_v: sub-Keplerian Speed
        St: Stokes number
        u_r: Gas accretion speed
        '''
                
        return -2*self.delta_v(r,t)/(self.St+self.St**-1)+self.u_r(r,t)/(1+self.St**2)
    
    def calc_lum(self,t):
        '''
        Calculate the luminosity of the star over time
        t can be float or array of time values
        if t is an array it must have shape (N_times,N_star) or (N_times)
        
        If 2D, will assume that times are equal for all stars
        
        returns data in shape (N_times,N_star)
        '''

        iso = pd.read_csv("mass_lum_isochrones.csv")
        isomass = iso["mass"].values        
        unique_isomass = np.unique(isomass)
        
        t = self.check_input_shape(t,return_1D=True)
        
        if type(t) is int or type(t) is float or type(t) is np.float64:
            
            lum_data = np.zeros((1,self.N_star))
        
        else:
            
            lum_data = np.zeros((len(t),self.N_star))
                 
        for i,star in enumerate(self.M_star/self.MSun2ME):

            diff = abs(unique_isomass-star)
            
            near_mass_idx = np.where(diff == np.min(diff))
            
            near_mass = unique_isomass[near_mass_idx]
            
            filt = isomass == near_mass
            
            iso_t = 10**iso["logt"].values[filt]
            iso_lum = 10**iso["lum"].values[filt]
            
            f = interpolate.interp1d(iso_t,iso_lum,bounds_error=False)
            
            lum_data[:,i] = f(t)

            lum_data[np.isnan(lum_data)] = f(min(iso_t))

        return lum_data
    
    def get_R1(self):
        '''
        Calculate initial disc size from integrating the surface density and 
        solving the analytical integral
        '''
        
        A = 39*self.kb*7.44*10**(-33)*self.alpha*150*self.calc_lum(0)**(2/7)*(self.M_star/self.MSun2ME)**(-1/7)*self.M_gas
        
        B = 28*self.M_0dot*np.sqrt(self.G*self.M_star)*self.mu_disc*self.mh*self.kg2ME

        R_1 =(self.r_in**(13/14)+A/B)**(14/13)
        
        return R_1
    
    def T_irr(self,r,t):
        
        r = self.check_input_shape(r)
        t = self.check_input_shape(t)
        
        T_irr = 150*(self.M_star.reshape(1,self.N_star)/self.MSun2ME)**(-1/7)*\
            self.calc_lum(t)**(2/7)*\
            (r)**(-self.zeta)
        
        return T_irr
    
    def T_visc(self,r,t):
        
        r = self.check_input_shape(r)
        t = self.check_input_shape(t)
        
        T_visc = 200*(self.M_star/self.MSun2ME)**(3/10)*\
            (self.alpha/10**-3)**(-1/5)*\
            (self.M_gdot(t)/self.MSun2ME/10**-8)**(2/5)*\
            r**(-9/10)
            
        return T_visc
    
    def T(self,r,t):
        
        T_irr = self.T_irr(r,t)
        T_visc = self.T_visc(r,t)
        
        if self.viscous_heating:
            
            return np.max(np.stack((T_irr,T_visc)),axis=0)
        
        return T_irr
    
    def M_gdot(self,t):
        
        t = self.check_input_shape(t)
        
        T1 = self.T_irr(self.R1,0)
        
        c_s1 = self.Mps2AUpYr*np.sqrt(self.kb*T1/(self.mu_disc*self.mh))
        nu1=self.alpha*c_s1**2/self.omega(self.R1)
        
        ts = 1/(3*(2-self.gamma)**2)*self.R1**2/nu1

        exp = -(5/2-self.gamma)/(2-self.gamma)
        
        return self.M_0dot*(t/ts+1)**exp
    
    def calc_rpf(self,t,return_derivative = False):
        
        r_pf = self.r_pf_init()
        
        try:
            
            len(t)
            
            times = t
            single = False

        except TypeError:
            
            times = np.linspace(1,t,100)
            single = True
    
        r_pf_vals = np.zeros((len(times),self.N_star))
    
        dr_pfdt_vals = np.zeros((len(times),self.N_star))
    
        dt = times[1]-times[0]
        
        for i,t_val in enumerate(times):

            T_rpf = self.T(r_pf,t_val)

            Z_rpf = self.ab_class.get_solid_frac(T_rpf)
        
            dr_pfdt = (2/3)*(3/16)**(1/3)*(self.G*self.M_star)**(1/3)*\
                      (self.eps_d*self.Z)**(2/3)*t_val**(-1/3) 
            
            if t_val == 0:
            
                dr_pfdt = np.zeros(self.N_star)
            
            r_pf_vals[i,:] = r_pf
            dr_pfdt_vals[i,:] = dr_pfdt
            
            r_pf+=dt*dr_pfdt
            
            if i != len(times)-1:
                
                dt = times[i+1]-times[i]
            
        if return_derivative:
            
            if single:
                
                return r_pf_vals[-1].reshape(1,self.N_star),dr_pfdt_vals[-1].reshape(1,self.N_star)
            
            return r_pf_vals,dr_pfdt_vals
        
        if single:
            
            return r_pf_vals[-1].reshape(1,self.N_star)
        
        return r_pf_vals
    
    def M_pdot(self,t,r=None):
        
        M_gdot = self.M_gdot(t)

        rpf,drpfdt = self.calc_rpf(t,return_derivative=True)
        
        u_rpf = self.u_r(rpf,t)
       
        if r is None:
    
            T_rpf = self.T(rpf,t)
            
            Z = self.ab_class.get_solid_frac(T_rpf)
            
            M_pdot = Z*M_gdot/-u_rpf*drpfdt   
            #M_pdot = self.Z*self.sigma_g(rpf,t)*2*np.pi*rpf*drpfdt
        
        else:
            
            r = self.check_input_shape(r)
            
            T = self.T(r,t)
            
            Z = self.ab_class.get_solid_frac(T)
            
            u_r = self.u_r(r,t)

            M_pdot = Z*M_gdot/-u_rpf*drpfdt
            
            #M_pdot = self.Z*self.sigma_g(rpf,t)*2*np.pi*rpf*drpfdt
            
            M_pdot[r>rpf] = 0
        
        return M_pdot
    
    def sigma_p(self,r,t):
        '''
        Surface density of pebbles
        '''
        r = self.check_input_shape(r)
        M_pdot = self.M_pdot(t,r)

        #Rdisc = self.Rdisc(t)
        vr = self.v_r(r,t)

        return M_pdot/(-2*np.pi*r*vr)
    
    def M_iso(self,r,t):
        '''
        Pebble isolation mass from Bitsch et al.(2018)

        Parameters
        ----------
        r: Radial location
        
        '''
        r = self.check_input_shape(r)
        frac1 = ((self.H(r,t)/r)/0.05)**3
        frac2 = (0.34*(np.log10(self.alpha_3)/np.log10(self.alpha_v))**4+0.66)
        frac3 = 1-((-self.chi+2.5)/6)

        return 25*frac1*frac2*frac3
    
    def sigma_g(self,r,t):
        '''
        Gas surface density
        '''
        r = self.check_input_shape(r)
        M_gdot = -self.M_gdot(t)
        
        return M_gdot/(2*np.pi*r*self.u_r(r,t))
    
    def rdot_type1(self,M,r,t):
        '''
        Time derivative of planet position
        
        Parameters
        ----------
        M: Mass of Planet [M_E]
        r: Radial location of planet [AU]
        M_gdot: Gas influx onto star [M_E yr^-1]

        Returns
        -------

        '''
        
        M = self.check_input_shape(M)
        r = self.check_input_shape(r)
        
        return -self.kmig*M/self.M_star*(self.sigma_g(r, t)*r**2/self.M_star)*(self.H(r,t)/r)**(-2)*self.v_k(r)
    
    def rdot(self,M,r,t):
        '''
        Modified Planet migration to take into account 
        how the migration rate changes after the gap is created
        
        '''
        M = self.check_input_shape(M)
        return self.rdot_type1(M,r,t)/(1+(M/(2.3*self.M_iso(r,t)))**2)
    
    def K(self,M,r,t):
        
        M = self.check_input_shape(M)
        r = self.check_input_shape(r)
        
        return (M/self.M_star)**2*(self.H(r,t)/r)**(-5)*self.alpha_v**(-1)
    
    def disc_acc(self,M,r,t):
        
        M = self.check_input_shape(M)
        r = self.check_input_shape(r)
        
        M_gdot = self.M_gdot(t)
        
        frac1 = (self.H(r,t)/r)**(-4)
        frac2 = (M/self.M_star)**(4/3)
        frac3 = M_gdot/self.alpha
        frac4 = 1/(1+0.04*self.K(M,r,t))
        dmdt = 0.29/(3*np.pi)*frac1*frac2*frac3*frac4
        #print(dmdt)
        #M_gap=np.sqrt(1/0.04*(self.H(r,t)/r)**5*self.alpha_v)*self.M_star
        #print(0.29/(3*np.pi)*frac1*frac2*frac3*1/(1+(M/M_gap)**2))
        
        return dmdt
        
    def KH_acc(self,M):
        '''
        Kelvin-Helmholtz-like contraction of the gas envelope
        Kappa is the envelope opacity [m^2 kg^-1]
        
        Parameters
        ----------
        M: Planet Mass [M_E]

        '''
        
        M = self.check_input_shape(M)
        
        return 10**(-5)*(M/10)**(4)*(self.kappa/0.1)**(-1)

    def R_B(self,M,r,t):
        
        M = self.check_input_shape(M)
        
        return self.G*M/self.c_s(r, t)**2

    def rho(self,M_tot,M_solids):
        
        N_bodies,N_star = M_tot.shape
        
        #Extend species properties so they cover all bodies and all stars
        density = np.repeat(self.density[np.newaxis,:],N_bodies,axis=0)
        density = np.repeat(density[:,np.newaxis,:],N_star,axis=1)
        v_i = M_solids/density
        vfrac = v_i/np.sum(v_i,axis = 2,keepdims=True)
        
        return np.sum(density*vfrac,axis=2)*5.61*10**8 #M_E/AU^3

    def T_rcb(self,M,r,t,L_pl):
        
        nabla_0 = self.nabla_0(M,r, t, L_pl)

        P_rcb = self.P_rcb(r,t,nabla_0)
        T_disc = self.T(r,t)
        P_disc = self.P(r,t)

        nabla_inf = 1/(4-self.beta_opacity)

        T_rcb = T_disc*(nabla_0/nabla_inf*(P_rcb/P_disc-1)+1)**nabla_inf
        
        return T_rcb
        
    def P_rcb(self,r,t,nabla_0):
        
        nabla_ad = (self.gamma_ad-1)/self.gamma_ad
        nabla_inf = 1/(4-self.beta_opacity)
        P_disc = self.P(r,t)
        
        return P_disc*(nabla_ad/nabla_0-nabla_ad/nabla_inf)/(1-nabla_ad/nabla_inf)
        
    def nabla_0(self,M,r,t,L_pl):
        
        M = self.check_input_shape(M)
        L_pl = self.check_input_shape(L_pl)
        
        T_disc = self.T(r,t)
        P_disc = self.P(r,t)
        
        return 3*self.kappa*P_disc*L_pl/(64*np.pi*self.G*M*self.sigma_sb*T_disc**4)
    
    def R_pl(self,M,rho):
        
        M = self.check_input_shape(M)
        rho = self.check_input_shape(rho)
        
        return (3*M/(4*np.pi*rho))**(1/3)
    
    def L_pl(self,M_dot,M,R):
        
        M = self.check_input_shape(M)
        R = self.check_input_shape(R)
        
        return self.G*M_dot*M/R
    
    def P(self,r,t):
        '''
        Comes from using midplane density in the ideal gas equation
        
        P = n*kb*T
        n = N/V
        N = m/(mu*m_H) => n = m/(mu*m_H*V)
        rho = m/V => n = rho/(mu*m_H)
        
        P = rho*kb*T/(mu*m_H)

        where rho is the midplane density        
        '''
                
        return self.rho_m(r,t)*self.kb*7.44*10**(-33)*self.T(r,t)/(self.mu_disc*self.mh*1.674*10**-25)
            
    def rho_m(self,r,t):
        
        return self.sigma_g(r,t)/(np.sqrt(2*np.pi)*self.H(r,t))
    
    def get_disc_lifetime(self):
        
        t_max = 10**7
        dt_max = 10**3
        #times = np.logspace(0,np.log10(t_max),100)
        #times = times[np.insert(np.diff(times),0,0)<dt_max]
        #times = np.append(times,np.arange(np.max(times)+dt_max,t_max,dt_max))
        
        times = np.arange(0,t_max+1,dt_max)
        
        dt = np.insert(np.diff(times),0,times[0])
        dt = np.repeat(dt[:,np.newaxis],self.N_star,axis=1)
        
        M_gdot = self.M_gdot(times)
        M_g_acc = np.cumsum((M_gdot+self.M_gphoto()*self.MSun2ME)*dt,axis=0)

        nearest_value = np.argmin(np.abs(M_g_acc-self.M_gas),axis=0)

        return times[nearest_value]

    def M_pdot_joanna(self,r,t,store_St = False):
        
        R1 = self.R1
        
        r_out = np.linspace(r,R1[0],1000)
        T_out0 = self.T(r_out,0)

        Z_out0 = self.ab_class.get_solid_frac(T_out0.reshape(1000,1))
        sigma_dust_out0 = Z_out0*self.sigma_g(r_out, 0)
        
        M_dust0_out = integrate.simps(sigma_dust_out0*2*np.pi*r_out,r_out,axis=0)

        M_dusti_out = np.copy(M_dust0_out)
        dt = t[1]-t[0]
        M_pdot = np.zeros((len(t),self.N_star))
        
        T = self.T(r,t)
        Z = self.ab_class.get_solid_frac(T)
        #print(Z/self.Z)
        
        #r_out = np.max((self.R1[0],1.5*r))
        #r_in = np.max((self.r_in,0.5*r))
        #dr = r_out-r_in
        
        if store_St:
            
            St_vals = np.zeros((len(t),self.N_star))
        
        for i,t_i in enumerate(t):
            
            sigma_g = self.sigma_g(r,t_i)
            T = self.T(r,t_i)
            Z = self.ab_class.get_solid_frac(T)
            sigma_dust = Z*sigma_g*M_dusti_out/M_dust0_out
            omega= self.omega(r)
            v_k = self.v_k(r)
            t_growth = sigma_g/sigma_dust*1/omega*(self.alpha_v/10**-4)**(-1/3)*(r)**(1/3)
            #rho = self.rho_m(r,t_i)
            #rho_in = self.rho_m(r_in,t_i)
            #rho_out = self.rho_m(r_out,t_i)
            #gp_in = rho_in*self.c_s(r_in,t_i)**2
            #gp_out = rho_out*self.c_s(r_out,t_i)**2
            vn=self.delta_v(r, t_i)
            
            #eta = (gp_out-gp_in)/dr/(2*rho*omega**2*r)
            #vn = abs(eta)*v_k

            f_dg = 30
        
            St0 = (np.sqrt(2*np.pi)*6.685*10**-18*5.61*10**11*1.25/sigma_g)
            v_f = self.v_f*self.Mps2AUpYr
        
            Stini = St0*np.exp(t_i/t_growth)
            Stf = (v_f**2/(3*self.alpha_v*self.c_s(r,t_i)**2))
            Stdf = (v_f/(2*vn))
            Stdrift = 1/(f_dg*vn/v_k)*sigma_dust/sigma_g
            St = np.min(np.stack((Stini,Stf,Stdf,Stdrift)),axis = 0)
            
            St[St<St0] = St0[St<St0]
        
            if store_St:
                    
                St_vals[i] = St
        
            v_r0 = 2*vn*St/(1+St**2)+abs(self.u_r(r,t_i)/(1+St**2))

            v_r = np.min(np.stack((v_r0,r/(f_dg*t_growth))),axis=0)
            
            M_pdoti = 2*np.pi*r*v_r*sigma_dust
            
            M_pdot[i] = M_pdoti

            M_dusti_out -= M_pdoti[0]*dt
            
            if i != len(t)-1:
                
                dt = t[i+1]-t[i]
             
        if store_St:
            
            return M_pdot,St_vals
        
        return M_pdot
    
    def sigma_p_joanna(self,r,t):
        
        eps_p = 0.5
        r = self.check_input_shape(r)
        t = self.check_input_shape(t)
        
        M_pdot = self.M_pdot_joanna(r, t)
        vk = self.v_k(r)
        sigma_g = self.sigma_g(r, t)
        
        return np.sqrt(2*M_pdot*sigma_g/(np.sqrt(3)*np.pi*eps_p*r*vk))
        
    
    def t_growth(self,r,t):
        
        T = self.T(r,t)

        Z = self.ab_class.get_solid_frac(T)
        
        omega = self.omega(r)
        
        return 1/Z*1/omega*(self.alpha_v/10**-4)**(-1/3)*(r)**(1/3)