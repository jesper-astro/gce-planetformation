# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:18:55 2022

@author: Jesper Nielsen
"""

import pandas as pd
import numpy as np

class Abundance:
    
    def __init__(self,init_abun,free_iron = False,carbon_grain=True):
        
        self.free_iron = free_iron
        
        self.carbon_grain = carbon_grain
        
        #Load the species data
        self.spec_data  = pd.read_csv("species_data.csv",delimiter = ";")
        
        #For output
        self.spec_names = self.spec_data["species"].values
        
        #For generating arrays
        self.N_species = len(self.spec_data)
        
        self.N_star = init_abun.shape[0]
        
        #Temperatures
        self.temps = self.spec_data["temp"].values
                
        self.init_abun=init_abun
        
        #Constants needed for calculations
        self.mu = self.spec_data["mass"].values
        
        #Element data
        self.el_data = pd.read_csv("element_data.csv",delimiter = ";")
        
        #Element names
        self.el_names = self.el_data["Element"]
                
        #Element mass
        self.el_mass = self.el_data['Mass']
        
        self.N_elements = len(self.el_data)
        
    def M_species(self,M_acc,T,gas = None):
        
        '''
        Mass accreted in one step
        
        Parameters
        ----------
        M_acc: Mass accreted during that time step [M_E]
        
        T: Temperature at the location of the planet
        
        gas (bool): If true, the planet is only accreting gas (i.e has reached 
                    pebble isolation mass), otherwise it is accreting pebbles
                            
        '''
        
        N_bodies = M_acc.shape[0]
        N_star = M_acc.shape[1]
        
        #Extend planet properties so that they cover all species
        #The data is of the form (Body, Species, Stars)
        #Each protoplanet around each star has a unique temperature
        #i.e. T is of the form (Body, Stars)

        T_ext = np.repeat(T[:,:,np.newaxis],self.N_species,axis=2)
        M_acc_ext = np.repeat(M_acc[:,:,np.newaxis],self.N_species,axis=2)
        
        vfrac_dict = self.get_vol_frac(T)
        vfrac = np.moveaxis(np.array(list(vfrac_dict.values())),0,-1)
        
        #Extend species properties so they cover all bodies and all stars
        mu = np.repeat(self.mu[np.newaxis,:],N_bodies,axis=0)
        mu = np.repeat(mu[:,np.newaxis,:],N_star,axis=1)
        
        temps = np.repeat(self.temps[np.newaxis,:],N_bodies,axis=0)
        temps = np.repeat(temps[:,np.newaxis,:],N_star,axis=1)
        
        #Used to find species which are gaseous
        #Each body has a specific temperature which means each
        #row is a unique temperature. This is copied for all species (columns)
        #and then masked
        T_filt = (T_ext > temps) & (temps != 0)

        #If the body is accreting gas, only count species which are gaseous
        #i.e if their condensation temperature is lower than the temperature of the gas
        #Otherwise, only accrete solids i.e. the reverse
                
        if gas is not None:
            
            if type(gas) != bool:
                
                raise TypeError("gas needs to be bool")
            
            if gas:
                
                vfrac[~T_filt] = 0
                m_spec_acc = mu*vfrac/np.sum(mu*vfrac,axis=2,keepdims=True)*M_acc_ext
                
                nan_filt = np.sum(mu*vfrac,axis=2) == 0
                m_spec_acc[nan_filt,:] = 0

                return m_spec_acc
            
            else:

                vfrac[T_filt] = 0
                m_spec_acc = mu*vfrac/np.sum(mu*vfrac,axis=2,keepdims=True)*M_acc_ext

                nan_filt = np.sum(mu*vfrac,axis=2) == 0

                m_spec_acc[nan_filt,:] = 0
                
                return m_spec_acc
        
        vfrac_solid = np.copy(vfrac)
        vfrac_gas = np.copy(vfrac)
        
        vfrac_solid[T_filt] = 0
        vfrac_gas[~T_filt] = 0
        
        m_spec_acc_solid = mu*vfrac_solid/np.sum(mu*vfrac_solid,axis=2,keepdims=True)*M_acc_ext
        m_spec_acc_gas = mu*vfrac_gas/np.sum(mu*vfrac_gas,axis=2,keepdims=True)*M_acc_ext        
        
        return m_spec_acc_solid,m_spec_acc_gas

    def get_vol_frac(self,T):
        
        '''
        Take in temperatures in shape of (N_bodies,N_star)
                              
        '''
        
        N_bodies,N_star = T.shape
        
        el_abun = self.init_abun
        
        #Need to reshape the abundances to be the same shape as the temperature
        C = el_abun["C/H"].values
        Fe = el_abun["Fe/H"].values
        N = el_abun["N/H"].values
        Si = el_abun["Si/H"].values
        S = el_abun["S/H"].values
        O = el_abun["O/H"].values
        Mg = el_abun["Mg/H"].values
        He = el_abun["He/H"].values
        
        C = np.repeat(C[np.newaxis,:],N_bodies,axis=0)
        Fe = np.repeat(Fe[np.newaxis,:],N_bodies,axis=0)
        N = np.repeat(N[np.newaxis,:],N_bodies,axis=0)
        Si = np.repeat(Si[np.newaxis,:],N_bodies,axis=0)
        S = np.repeat(S[np.newaxis,:],N_bodies,axis=0)
        O = np.repeat(O[np.newaxis,:],N_bodies,axis=0)
        Mg = np.repeat(Mg[np.newaxis,:],N_bodies,axis=0)
        He = np.repeat(He[np.newaxis,:],N_bodies,axis=0)
        
        N_species = len(self.spec_names)        

        vfracs = dict(zip(self.spec_names,np.zeros((N_species,N_bodies,N_star))))
        
        #Constant throughout the disc
        #Throughout the following sections, I use np.min(np.stack())
        #This is because I don't want to accidentally use more of an abundance
        #when calculating the volume fractions
        
        #For example, if C/H=10*O/H and I set CO=0.45C, I will use more oxygen
        #than I have. Therefore, I have to consider the minimum in each species
        
        #Use np.stack() to join to arrays along a new axis and calculate the
        #minimum on that axis
        
        if self.carbon_grain:
            
            vfracs["C-O"] = np.min(np.stack((0.65*C,O)),axis=0)
            vfracs["C-O2"] = np.min(np.stack((0.2*C,(O-vfracs["C-O"])/2)),axis=0)
            vfracs['C'] = (C-vfracs['C-O']-vfracs['C-O2']-vfracs['C-H4'])
        
        else:
            
            T_filtC = T<=70
            vfracs["C-O"][T_filtC] = np.min(np.stack((0.9*C,O)),axis=0)[T_filtC]
            vfracs["C-O2"][T_filtC] = np.min(np.stack((0.1*C,(O-vfracs["C-O"])/2)),axis=0)[T_filtC]
            vfracs["C-H4"][T_filtC] = (C-vfracs["C-O"]-vfracs["C-O2"])[T_filtC]
        
            vfracs["C-O"][~T_filtC] = np.min(np.stack((0.45*C,O)),axis=0)[~T_filtC]
            vfracs["C-O2"][~T_filtC] = np.min(np.stack((0.1*C,(O-vfracs["C-O"])/2)),axis=0)[~T_filtC]
            vfracs["C-H4"][~T_filtC] = (C-vfracs["C-O"]-vfracs["C-O2"])[~T_filtC]
        
        #Base initial abundances #############
        vfracs["N-H3"]= 0.1*N
        vfracs["N2"] = 0.9*N/2
        
        vfracs["He"] = He
        
        vfracs["Fe"] = Fe
        vfracs["Si"] = Si
        vfracs["Mg"] = Mg
        
        vfracs["H2-S"] = S
        ########################################
        
        #Form SiO
        vfracs["Si-O"] = np.min(np.stack((Si,O-vfracs["C-O"]-2*vfracs["C-O2"])),axis=0)
        vfracs['Si'] -= vfracs['Si-O']
        
        #Baselines of water and H2
        vfracs["H2-O"] = (O-vfracs["C-O"]-2*vfracs["C-O2"]-vfracs["Si-O"])
        
        vfracs["H2"] = (1-2*vfracs["H2-O"]-4*vfracs["C-H4"]-2*vfracs["H2-S"]-3*vfracs['N-H3'])/2

        #Mg2SiO4 (Forsterite) gets condensed from free Mg and SiO
        #2Mg + SiO + 3H2O = Mg2SiO4 + 3H2
        T_filt2 = (T<1354)

        vfracs["Mg2-Si-O4"][T_filt2] = np.min(np.stack((Mg/2,vfracs['Si-O'],vfracs['H2-O']/3)),axis = 0)[T_filt2]
        
        vfracs["Si-O"][T_filt2] -= vfracs["Mg2-Si-O4"][T_filt2]

        vfracs['Mg'][T_filt2] -= 2*vfracs['Mg2-Si-O4'][T_filt2]
        
        vfracs["H2-O"][T_filt2] -= 3*vfracs['Mg2-Si-O4'][T_filt2]
        
        vfracs['H2'][T_filt2] += 3*vfracs["Mg2-Si-O4"][T_filt2]
        
        
        #Enstatite is formed from forsterite reacting with water
        #Mg2SiO4 + SiO + H2O = 2MgSiO3 + H2
        #SiO + H2O = SiO2 + H2
        
        T_filt3 = (T<1316)
        
        vfracs['Mg-Si-O3'][T_filt3] = 2*np.min(np.stack((vfracs['Mg2-Si-O4'],vfracs['Si-O'],vfracs['H2-O'])),axis=0)[T_filt3]
        
        vfracs['H2-O'][T_filt3] -= 1/2*vfracs['Mg-Si-O3'][T_filt3]
        
        vfracs['Mg2-Si-O4'][T_filt3] -= 1/2*vfracs['Mg-Si-O3'][T_filt3]
        
        vfracs['Si-O'][T_filt3] -= 1/2*vfracs['Mg-Si-O3'][T_filt3]
        
        vfracs['H2'][T_filt3] += 1/2*vfracs['Mg-Si-O3'][T_filt3]
        
        #SiO2 formation
        vfracs['Si-O2'][T_filt3] = np.min(np.stack((vfracs['Si-O'],vfracs['H2-O'])),axis=0)[T_filt3]
        
        vfracs['Si-O'][T_filt3] -= vfracs['Si-O2'][T_filt3]
        
        vfracs['H2'][T_filt3] += vfracs['Si-O2'][T_filt3]
        
        vfracs['H2-O'][T_filt3] -= vfracs['Si-O2'][T_filt3]

        #FeS form from the corrosion of Fe alloy (free iron) by H2S
        #H2S + Fe = FeS + H2
        
        T_filt4 = (T<710)
        
        vfracs["Fe-S"][T_filt4] = np.min(np.stack((vfracs['Fe'],vfracs['H2-S'])),axis=0)[T_filt4]
        vfracs["H2-S"][T_filt4] -= vfracs['Fe-S'][T_filt4]
        vfracs['Fe'][T_filt4] -= vfracs['Fe-S'][T_filt4]
        vfracs['H2'][T_filt4] += vfracs['Fe-S'][T_filt4]
        
        #Fayalite (Fe2SiO4) gets formed from Enstatite, forms a forsterite in the process
        #Either use up all the remaining free iron or all enstatite
        #2Fe + 2MgSiO3 + 2H2O = Mg2SiO4 + Fe2SiO4 + H2
        
        T_filt5 = (T<480)

        if not self.free_iron:
        
            vfracs["Fe2-Si-O4"][T_filt5] = np.min(np.stack((vfracs['Fe']/2,vfracs["Mg-Si-O3"]/2,vfracs["H2-O"]/2)),axis = 0)[T_filt5]
        
            vfracs["Mg2-Si-O4"][T_filt5] += vfracs["Fe2-Si-O4"][T_filt5]
            vfracs["Mg-Si-O3"][T_filt5] -= 2*vfracs["Fe2-Si-O4"][T_filt5]
            vfracs["H2-O"][T_filt5] -= 2*vfracs["Fe2-Si-O4"][T_filt5]
            vfracs['H2'][T_filt5] += 2*vfracs['Fe2-Si-O4'][T_filt5]
            vfracs['Fe'][T_filt5] -= 2*vfracs['Fe2-Si-O4'][T_filt5]
        
        #If there is free iron left, Magnetite (Fe3O4) is formed from water vapour+Fe
        #Rest remain the same below 370K
        T_filt6 = (T<370)
                
        #3Fe+4H2O = Fe3O4+4H2
        if not self.free_iron:
            
            vfracs["Fe3-O4"][T_filt6] = np.min(np.stack((vfracs['Fe']/3,vfracs["H2-O"]/4)),axis = 0)[T_filt6]
            vfracs['H2-O'][T_filt6] -= 4*vfracs["Fe3-O4"][T_filt6]
            vfracs['Fe'][T_filt6] -= 3*vfracs["Fe3-O4"][T_filt6]
            vfracs['H2'][T_filt6] += 4*vfracs["Fe3-O4"][T_filt6]
        
        return vfracs

    def get_solid_frac(self,T):
        '''
        Calculate the metallicity at temperature T
        '''
        #Volume fraction of all species

        vfrac = self.get_vol_frac(T)
        
        N_bodies,N_star = T.shape
        
        el_names = self.el_names
        
        solid_frac = dict(zip(el_names,np.zeros((self.N_elements,N_bodies,N_star))))
        
        #Easiest to iterate over species names and pick out each element in them
        for i,spec in enumerate(self.spec_names):
            
            vfrac_spec = vfrac[spec]
            
            #Ignore gases
            gas = (self.temps[i] != 0) & (T>self.temps[i])
            gas = gas[:,0]
            
            vfrac_spec[gas] = 0
            
            #Find element names
            split_name = spec.split("-")
            for el_name in split_name:
                
                if len(el_name) == 2:
                    #Either e.g. Fe or e.g. H2
                    #Try to convert second character to int, if doesn't work
                    #it's an element with two letters
                    try:
                    
                        el_nr = int(el_name[1])
                        solid_frac[el_name[0]] += el_nr*vfrac_spec
                        
                    except ValueError:
                        
                        solid_frac[el_name]+=vfrac_spec
                
                #Single character elements e.g. S
                elif len(el_name) == 1:
    
                    solid_frac[el_name]+=vfrac_spec
                
                #Double character elements with multiple atoms e.g. Fe2
                elif len(el_name) == 3:
                    
                    solid_frac[el_name[:2]] += int(el_name[2])*vfrac_spec

        solid_massfrac = np.zeros((N_bodies,N_star))
        
        #Go through each element and calculate their contribution to Z
        for i,el in enumerate(solid_frac.keys()):
            
            if el == 'H' or el == 'He':
            
                continue
            
            solid_massfrac += self.el_mass.values[i]*solid_frac[el]/(1+4*self.init_abun['He/H'].values)
        
        return solid_massfrac

    def find_Z(self):
        
        #Find the metallicity given that all heavy elements are solid.
        Z = np.zeros(self.N_star)

        for i,el in enumerate(self.init_abun.columns):
            el_name=el.split("/")[0]
            if el_name == "He":
                continue
            idx = np.where(self.el_data["Element"]==el_name)[0][0]

            Z+=self.el_data["Mass"].values[idx]*self.init_abun.values[:,i]/(1+self.init_abun["He/H"].values*4)
            
        return Z
    
    def get_mmw(self,T):
        
        '''
        Calculate the mean molecular weight of gas
        '''
        
        N_bodies,N_star = T.shape
        
        vfrac = self.get_vol_frac(T)
        vfrac_vals = np.moveaxis(np.array(list(vfrac.values())),0,-1)
        
        mu_full = np.repeat(self.mu[np.newaxis,:],N_bodies,axis=0)
        mu_full = np.repeat(mu_full[:,np.newaxis,:],N_star,axis=1)
        
        temps = np.repeat(self.temps[np.newaxis,:],N_bodies,axis=0)
        temps = np.repeat(temps[:,np.newaxis,:],N_star,axis=1)
        
        T_ext = np.repeat(T[:,:,np.newaxis],self.N_species,axis=2)

        full_T_filt = (T_ext > temps) & (temps != 0)
        
        vfrac_vals[~full_T_filt] = 0
        
        norm_gas = np.sum(mu_full*vfrac_vals,axis=2,keepdims = True)
        
        return 1/(np.sum(vfrac_vals/norm_gas,axis=2))
        
    def get_mass_frac(self,T,name=None,form='solid'):
        
        if name is None:
            keepdims=True
        else:
            keepdims=False
        
        if form not in ['solid','gas']:
            
            raise ValueError('Form must be solid or gas')
        
        N_bodies = T.shape[0]
        
        vol_frac = self.get_vol_frac(T)
        vfrac_vals = np.moveaxis(np.array(list(vol_frac.values())),0,-1)

        mu = np.repeat(self.mu[np.newaxis,:],N_bodies,axis=0)
        mu = np.repeat(mu[:,np.newaxis,:],self.N_star,axis=1)

        temps = np.repeat(self.temps[np.newaxis,:],N_bodies,axis=0)
        temps = np.repeat(temps[:,np.newaxis,:],self.N_star,axis=1)

        T = np.repeat(T[:,:,np.newaxis],self.N_species,axis=2)

        T_filt = (T > temps) & (temps != 0)

        vfrac_solid = np.copy(vfrac_vals)
        vfrac_gas = np.copy(vfrac_vals)

        if form == 'solid':
            
            vfrac_vals[T_filt] = 0
            norm = np.sum(mu*vfrac_vals,axis=2,keepdims=keepdims)
        
        elif form == 'gas':
            
            vfrac_vals[~T_filt] = 0
            norm = np.sum(mu*vfrac_vals,axis=2,keepdims=keepdims)
            
        if name is not None:
            
            spec_idx = np.where(self.spec_data["species"] == name)[0][0]
            mu_spec = self.spec_data['mass'].loc[spec_idx]

            return vfrac_vals[:,:,spec_idx]*mu_spec/norm

        return dict(zip(self.spec_names,np.moveaxis(mu*vfrac_vals/norm,-1,0)))   

def test(stars,free_iron=False,carbon_grain=True):
    
    '''
    Inputs some stellar data in the same form as whe integrating
    Headers need to have abundance names in the form X/H
    Calculates the number abundances for relevant temperatures and 
    check if the sums add up to the initial elemental abundance
    '''
    
    N_star = len(stars)
    
    try:
        stars_in = stars.drop(columns=["name","loglum","mass"])
    except KeyError:
        stars_in = stars
    
    vfracs_out = []
    
    for i in range(N_star):
        
        star = stars_in[i:i+1]

        A = Abundance(star,free_iron=free_iron,carbon_grain=carbon_grain)
        T = np.linspace(20,1420,20).reshape(20,1)
    
        vfracs=A.get_vol_frac(T)
        C_spec=vfracs["C-O"]+vfracs["C-O2"]+vfracs["C-H4"]+vfracs['C']
        Si_spec=vfracs["Si"]+vfracs["Si-O"]+vfracs["Mg-Si-O3"]+vfracs["Mg2-Si-O4"]+vfracs["Fe2-Si-O4"]+vfracs["Si-O2"]
        O_spec=vfracs["C-O"]+2*vfracs["C-O2"]+vfracs["H2-O"]+vfracs["Si-O"]+2*vfracs["Si-O2"]+4*vfracs["Fe3-O4"]+4*vfracs["Mg2-Si-O4"]+4*vfracs["Fe2-Si-O4"]+3*vfracs["Mg-Si-O3"]
        Mg_spec=vfracs["Mg"]+vfracs["Mg-Si-O3"]+2*vfracs["Mg2-Si-O4"]
        Fe_spec=vfracs["Fe"]+vfracs["Fe-S"]+3*vfracs["Fe3-O4"]+2*vfracs["Fe2-Si-O4"]
        
        if np.any(C_spec - star["C/H"].values < -10**-16) or np.any(C_spec-star['C/H'].values > 10**-16):
            print(i,"C")
        if np.any(Si_spec - star["Si/H"].values < -10**-16) or np.any(Si_spec-star['Si/H'].values > 10**-16):
            print(i,"Si")
        if np.any(O_spec - star["O/H"].values < -10**-16) or np.any(O_spec-star['O/H'].values > 10**-16):
            print(i,"O")
        if np.any(Mg_spec - star["Mg/H"].values < -10**-16) or np.any(Mg_spec-star['Mg/H'].values > 10**-16):
            print(i,"Mg")
        if np.any(Fe_spec - star["Fe/H"].values < -10**-16) or np.any(Fe_spec-star['Fe/H'].values > 10**-16):
            print(i,"Fe")
        
        vfracs_out.append(vfracs)
    
    return vfracs_out
        