# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:42:44 2022

@author: Jesper Nielsen
"""
import numpy as np
#from massgrowth import GrowthTrack as GT
import h5py
import os
from integrate_viscous import IntegrateJoanna as Int
import pandas as pd
import func

np.random.seed(1)
calc=func.Calculator

#######################################################
#Constants

MS2ME = 1.9891*10**30/(5.9722*10**24)
Mps2AUpYr = 2.108*10**-4 
M_sun = 1*MS2ME

theta_names = ["St","Z","alpha","M_star","zeta","beta","kappa","delta","v_f"]
#######################################################

#Input parameters

dtypes = {"N_grid":int,"N_bodies":int,
          "output_name":str,"sun":bool,
          "overwrite":bool,"store_full":bool,
          "iso":str,"free_iron":bool,'carbon_grain':bool,
          "evap":bool,"grid":bool,
          "t_low":float,"t_high":float,"r_low":float,
          "r_high":float,"dt":float,"M_0":float,"t_end":float}

args = pd.read_csv("init_args.csv",dtype=dtypes,delimiter=";")

N_grid = args["N_grid"].loc[0]
N_bodies = args["N_bodies"].loc[0]

output_name = args["output_name"].loc[0]
input_name = args["input_name"].loc[0]
sun = args["sun"].loc[0]

overwrite = args["overwrite"].loc[0]
store_full = args["store_full"].loc[0]

#Allowed values for iso: bitsch, gap

#Sets how I want to find isolation mass M_gap/2.3 or from bitsch
iso = args["iso"].loc[0]

#Sets if I only have free iron (apart from FeS) or 
#if I include Magnetite and Fayalite
free_iron = args["free_iron"].loc[0]

#If I want to have carbon grains in the model
carbon_grain = args['carbon_grain'].loc[0]

#If I want to switch to a constant flux ratio
constant_flux = args['constant_flux'].loc[0]

viscous_heating = args['viscous_heating'].loc[0]

#Include evaporation of species during accretion
evap = args["evap"].loc[0]

#Sets whether or not I integrate many bodies in a grid5
#or if I set thepositions and starting times myself
grid = args["grid"].loc[0]

#Lower limit for starting times of embryos
t_low = args["t_low"].loc[0]*10**6

#Lower limit for starting positions of embryos
r_low = args["r_low"].loc[0]

#Timestep
dt = args["dt"].loc[0]*10**6

St = args['St'].loc[0]
delta = args['delta'].loc[0]
v_f = args['v_f'].loc[0]

#Initial masses of embryos
m_0 = args["M_0"].loc[0]

if sun:

    init_abun = pd.read_csv("./solar_abun.csv")
    
    star_names = np.array(["sun"]).astype("S")
    N_star = 1

    zeta = np.array([3/7])
    beta = np.array([15/14])
    alpha = np.array([0.01])
    St_val = np.array([St])
    M_star = np.array([M_sun])
    kappa = np.array([0.005])
    delta_val = np.array([delta])
    v_f_val = np.array([v_f])

else:
    
    init_abun = pd.read_csv("./{}.csv".format(input_name))

    star_names = init_abun["name"].values.astype("S")
    N_star = len(init_abun)
    init_abun.drop(columns = "name",inplace=True)
    
    zeta = np.repeat(3/7,N_star)
    beta = np.repeat(15/14,N_star)
    alpha = np.repeat(0.01,N_star)
    St_val = np.repeat(St,N_star)
    M_star = init_abun["mass"].values*MS2ME
    kappa = np.repeat(0.005,N_star)
    delta_val = np.repeat(delta,N_star)
    v_f_val = np.repeat(v_f,N_star)
    
    try:
        
        init_abun.drop(columns = ["mass","lum"],inplace=True)
    
    except KeyError:
        
        init_abun.drop(columns = ["mass"],inplace=True)
        
Z = np.zeros(N_star)

el_data = pd.read_csv("element_data.csv",delimiter = ";")

for i,el in enumerate(init_abun.columns):
    el_name=el.split("/")[0]
    if el_name == "He":
        continue
    idx = np.where(el_data["Element"]==el_name)[0][0]
    Z+=el_data["Mass"].values[idx]*init_abun.values[:,i]/(1+init_abun["He/H"].values*4)

theta = np.array([St_val,Z,alpha,M_star,zeta,beta,kappa,delta_val,v_f_val])
track = calc(theta,abun=init_abun,viscous_heating=viscous_heating)

r_low = np.repeat(r_low,N_star)
r_high = 0.95*track.R1[0]

#End point for the disc
t_end = track.get_disc_lifetime()

t_high = 0.95*t_end

#Starttime for the integration
t0=0

#Initialise the input data
if grid:
    
    N_bodies = N_grid**2
    #Everything in here can be varied
    #Will only be used if I want to create maps over parameter space
    if len(t_high) > 1:

        starttimes = np.linspace(t_low,t_high,N_grid).T

    else:
        
        starttimes = np.ones((N_star,N_grid))*np.linspace(t_low,t_high,N_grid).T

    r_in = np.ones((N_star,N_grid))*np.linspace(r_low,r_high,N_grid)

    m_in = np.ones((N_star,N_bodies))*m_0
    
else:
    
    #Here the difference is that I don't necessary initialise the positions
    #and times in set intervals.

    m_in = np.ones((N_star,N_bodies))*m_0
    
    r_in = 10**np.random.uniform(np.log10(r_low),np.log10(r_high),(N_bodies,N_star)).T
    starttimes = np.random.uniform(t_low,t_high,(N_bodies,N_star))

if theta.shape[1] != N_star:
    
    raise ValueError("Number of inputted stars does not match number of stellar parameters in model")

if grid:
    
    starttimes = np.repeat(starttimes,N_grid,axis = 1).T
    r_in = np.tile(r_in,N_grid)
    
data = np.array([m_in,r_in]).swapaxes(0,2)

#######################################################################
#Run the integration
#######################################################################

integrator = Int(theta,data,dt,init_abun,starttimes,t0,t_end,
                free_iron=free_iron,carbon_grain=carbon_grain,
                store_full=store_full,evap=evap,constant_flux = constant_flux,
                viscous_heating=viscous_heating)

t_out,mr_out,gas_out,solid_out,names = integrator.integrate()

N_species = len(names)

#######################################################################
#Create output file
#######################################################################
output_dir = "data/"+output_name
output_file = os.path.join(output_dir,"output.h5")

if not os.path.exists(output_dir):
    
    os.makedirs(output_dir)
    print("Created output directory:" + output_dir)

#If it exists and I have not set overwrite to True, raise error
if os.path.exists(output_file):
    
    if not overwrite:
        
        raise IOError(output_file + " already exists, please try again")
    
    else:
        
        os.remove(output_file)

#Write the data to the file
with h5py.File(output_file,"a") as file:

    file.create_group("input")
    file.create_group("output")
    
    file["input"].create_dataset(name="star_names",data=star_names)
    
    file["input"].create_dataset("N_star",data=N_star)
    file["input"].create_dataset("N_species",data=N_species)
    file["input"].create_dataset("N_bodies",data=N_bodies)
    file["input"].create_dataset("r0_init",data=r_in)
    file["input"].create_dataset("t0_init",data=starttimes.T)
    file["input"].create_dataset("iso_type",data=iso)
    file['input'].create_dataset('viscous_heating',data=int(viscous_heating))
    file['input'].create_dataset('carbon_grain',data=int(carbon_grain))
    file['input'].create_dataset('free_iron',data=int(free_iron))
    file['input'].create_dataset('constant_flux',data=int(constant_flux))
    file['input'].create_dataset('min_wfrac',data=track.ab_class.get_mass_frac(np.ones((1,N_star))*69,name='H2-O')[0])
    file['input'].create_dataset('R1',data = integrator.R1[0])
    
    dtype_theta = np.dtype({"names":theta_names,"formats":[(float)]*len(theta)})
    
    rec_theta = np.rec.fromarrays(theta, dtype=dtype_theta)
    
    file["input"].create_dataset("theta",data=rec_theta)

    dtype_abun = np.dtype({"names":init_abun.columns.values,"formats":[(float)]*len(init_abun.columns)})
    
    rec_abun = np.rec.fromarrays(init_abun.values.T, dtype=dtype_abun)
    
    file["input"].create_dataset("init_abun",data=rec_abun)

    #If store_full is True, the output has an extra dimension for each timestep
    #otherwise, I don't store the data at every step
    
    #Store position and mass
    if store_full:
        
        file["output"].create_dataset("final_sma",data=mr_out[-1,:,:,1].T)
        file["output"].create_dataset("final_mass",data=mr_out[-1,:,:,0].T)
    
    else:
        
        file["output"].create_dataset("final_sma",data=mr_out[:,:,1].T)
        file["output"].create_dataset("final_mass",data=mr_out[:,:,0].T)
        
    file["output"].create_dataset("time",data=t_out)

    file["input"].create_dataset("species_names",data=names.astype("S"))
    
    #Again if the full data is stored, an axis is added
    
    if store_full:
    
        final_solid_mass_el,final_gas_mass_el = func.get_element_mass(gas_out[-1,:,:,:],solid_out[-1,:,:,:],names)
                
        #Go through each species                    
        for i_spec,spec_name in enumerate(names):
            
            if spec_name == "Fe" or spec_name == "Mg" or spec_name == "He" or spec_name == "Si" or spec_name == 'C':
                
                spec_name = spec_name+"_free"
            
            file["output"].create_dataset(name=spec_name,data=solid_out[-1,:,:,i_spec].T+gas_out[-1,:,:,i_spec].T)
            file["output"].create_dataset(name=spec_name+"_solid",data=solid_out[-1,:,:,i_spec].T)
            file["output"].create_dataset(name=spec_name+"_gas",data=gas_out[-1,:,:,i_spec].T)    
        
        for i_star,star_name in enumerate(star_names):

            file["output"].create_group(star_name)                

            file["output"][star_name].create_dataset("mass",data=mr_out[:,:,i_star,0])
            file["output"][star_name].create_dataset("sma",data=mr_out[:,:,i_star,1])
        
            file["output"][star_name].create_group("species_gas_full")
            file["output"][star_name].create_group("species_solids_full")
    
            for i_spec,name in enumerate(names):
        
                file["output"][star_name]["species_gas_full"].create_dataset(name,data=gas_out[:,:,i_star,i_spec])
                file["output"][star_name]["species_solids_full"].create_dataset(name,data=solid_out[:,:,i_star,i_spec])
        
        file['output'].create_dataset(name='final_solid_mass',data = np.sum(solid_out[-1],axis=2).T)
        file['output'].create_dataset(name='final_gas_mass',data = np.sum(gas_out[-1],axis=2).T)
        
    else:
        
        final_solid_mass_el,final_gas_mass_el = func.get_element_mass(gas_out,solid_out,names)
                    
        for i_spec,spec_name in enumerate(names):
            
            if spec_name == "Fe" or spec_name == "Mg" or spec_name == "He" or spec_name == "Si" or spec_name == "C":
                
                spec_name = spec_name+"_free"
            
            file["output"].create_dataset(name=spec_name,data=solid_out[:,:,i_spec].T+gas_out[:,:,i_spec].T)
            file["output"].create_dataset(name=spec_name+"_solid",data=solid_out[:,:,i_spec].T)
            file["output"].create_dataset(name=spec_name+"_gas",data=gas_out[:,:,i_spec].T)    
    
        file['output'].create_dataset(name='final_solid_mass',data = np.sum(solid_out,axis=2).T)
        file['output'].create_dataset(name='final_gas_mass',data = np.sum(gas_out,axis=2).T)

    el_names = np.array(list(final_solid_mass_el.keys()))
    solid_mass_el = np.array(list(final_solid_mass_el.values()))
    gas_mass_el = np.array(list(final_gas_mass_el.values()))

    file["input"].create_dataset("element_names",data=el_names.astype("S"))     
        
    for i_el,el_name in enumerate(el_names):
        
        file["output"].create_dataset(name=el_name,data=solid_mass_el[i_el,:,:]+gas_mass_el[i_el,:,:])
        file["output"].create_dataset(name=el_name+"_solid",data=solid_mass_el[i_el,:,:])
        file["output"].create_dataset(name=el_name+"_gas",data=gas_mass_el[i_el,:,:])
