# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 09:59:40 2022

@author: Jesper Nielsen
"""
import h5py
import numpy as np
from func import HZ
from func import lum
from statsmodels.stats.proportion import proportion_confint as conf
import pandas as pd

ME2MSun = 1/(1.9891*10**30/(5.9722*10**24))

class DataHandler:
    
    def __init__(self,filename):
        
        self.file = file = h5py.File(filename, mode = "r")
        
        self.output = output = file["output"]
        
        self.times = output["time"][()]
        
        self.input = file["input"]
        
        self.N_species = self.input["N_species"][()]
        self.N_elements = len(self.input["element_names"][()])
        self.N_bodies = self.input["N_bodies"][()]
        self.N_star = self.input["N_star"][()]
        self.spec_names = self.input["species_names"][()].astype(str)
        self.el_names = self.input["element_names"][()].astype(str)
        self.star_names = self.input["star_names"][()].astype(str)
        self.wfrac_lim = np.repeat(self.input['min_wfrac'][()][:,np.newaxis],self.N_bodies,axis=1)
        
        self.spec_data = pd.read_csv('species_data.csv',sep=';')
        self.el_data = pd.read_csv('element_data.csv',sep = ';')
        
        input_abun = self.input["init_abun"][()]
        
        self.abun_names = np.array(list(input_abun.dtype.names))
                
        self.input_abun = np.array([list(i) for i in input_abun])
        
        self.free_elements = ['Fe','Mg','He','Si','C']
        
        theta = self.input["theta"][()]
        self.theta_names = np.array(theta.dtype.names)
        
        if self.N_star == 1:
            
            self.theta = np.array(list(theta[0]))
        
        else:
            
            self.theta = np.array([list(i) for i in theta])
            
        self.r_init = self.input["r0_init"][()]
        self.t_init = self.input["t0_init"][()]
        
        #T_idx = np.where(self.theta_names == "T_eff")[0][0]
        M_star_idx = np.where(self.theta_names == "M_star")[0][0]
        
        if self.N_star == 1:
            
            self.M_star = self.theta[M_star_idx]
         
            #self.T_eff = self.theta[T_idx]
            
        else:
                            
            #self.T_eff = self.theta[:,T_idx]
                
            self.M_star = self.theta[:,M_star_idx]
    
    def get_mass(self,form = None,star_idx = None):
        
        type_list = ['gas','solid']
        
        if form is not None:
            
            if form in type_list:
                
                if star_idx is not None:
                    
                    return self.output['final_{}_mass'.format(form)][()][star_idx]
                
                return self.output['final_{}_mass'.format(form)][()]
            
            else:
                
                raise ValueError('Form must be gas, solid or None')
        
        if star_idx is not None:
            
            return self.output["final_mass"][()][star_idx]
        
        return self.output["final_mass"][()]
    
    def get_sma(self,star_idx = None):
        
        if star_idx is not None:
            
            return self.output["final_sma"][()][star_idx]
        
        return self.output["final_sma"][()]
 
    def get_species(self,name,form = None,star_idx = None):
        
        type_list = ['gas','solid']
        
        if form is not None:
            
            if form in type_list:
                
                if star_idx is not None:
                    
                    return self.output['{}_{}'.format(name,form)][()][star_idx]
                
                return self.output['{}_{}'.format(name,form)][()]
            
            else:
                
                raise ValueError('Form must be gas, solid or None')
        
        if star_idx is not None:
            
            return self.output[name][()][star_idx]
        
        return self.output[name][()]   
    
    def get_planet_radii(self,star_idx = None):
        
        rho = self.get_planet_density(star_idx)*0.04345 #M_E/R_E^3
        
        mass = self.output['final_solid_mass'][()]
        
        return (3*mass/(4*np.pi*rho))**(1/3)

    def get_planet_density(self,star_idx = None):
        
        MEarth = 5.97*10**24
        
        density = self.spec_data["density"].values
        density = np.repeat(density[np.newaxis,:],self.N_bodies,axis=0)
        density = np.repeat(density[np.newaxis,:,:],self.N_star,axis=0)
        
        spec_mass = np.zeros((self.N_star,self.N_bodies,self.N_species))
        
        for i_spec,spec_name in enumerate(self.spec_data['species'].values):

            if spec_name in self.free_elements:
                
                spec_mass[:,:,i_spec] = self.output[spec_name+'_free_solid'][()]
                
            else:
                
                spec_mass[:,:,i_spec] = self.output[spec_name+'_solid'][()]
        
        vi = spec_mass*MEarth/density
        
        vfrac = vi/np.sum(vi,axis = 2,keepdims=True)
        
        rho = np.sum(density*vfrac,axis=2)*0.001 #want it to be in g/cm^3?
        
        if star_idx is not None:
            
            return rho[star_idx]
    
        return rho
    
    def get_cmf(self,star_idx=None):
       
        m_Fe = self.output['Fe-S_solid'][()]+self.output['Fe_free_solid'][()]
        
        f_iron = m_Fe/(self.output['final_solid_mass'][()]-\
                       self.output['H2-O_solid']-\
                       self.output['C-O_solid']-self.output['C-O2_solid']-\
                        self.output['N2_solid']-self.output['N-H3_solid'])
        
        if star_idx is not None:
        
            return f_iron[star_idx]
        
        return f_iron
     
    def close(self):
        
        self.file.close()
        
        
        