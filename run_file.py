#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:10:34 2022

@author: jesperroslund
"""

import pandas as pd
import runpy
import numpy as np

def sun(grid = False,St=0.01,v_f=2,delta=0.0001,free_iron=True,carbon_grain=True,
        viscous_heating=False,constant_flux=False):
        
    init_args = pd.read_csv('init_args.csv',sep=';')
    out_name = 'sun_joanna'
    
    init_args['sun'] = 1
    init_args['St'] = St
    init_args['dt'] = 0.001
    init_args['r_low'] = 0.5
    init_args['t_low'] = 0.01
    init_args['N_bodies'] = 5*10**3
    init_args['sun'] = 1
    init_args['store_full'] = 0
    init_args['free_iron'] = int(free_iron)
    init_args['carbon_grain'] = int(carbon_grain)
    init_args['evap'] = 1
    init_args['overwrite'] = 1
    init_args['v_f']=v_f
    init_args['delta']=float(delta)
    init_args['viscous_heating']=int(viscous_heating)
    init_args['constant_flux']=int(constant_flux)

    if not grid:
        
        init_args['grid'] = 0
        
    elif grid:
                        
        out_name+='_grid'

        init_args['grid'] = 1
    
    if free_iron:
        
        out_name+='_freeiron'
    
    if not carbon_grain:
        
        out_name +='_nograin'
        
    if constant_flux:
        
        out_name +='_constantflux'
    
    if v_f != 2:
        
        out_name+='_vf{}'.format(str(v_f))
    
    if delta != 0.0001:
        
        out_name+='_delta{}'.format(str(delta).split('.')[1])
    
    init_args['output_name'] = out_name
    
    init_args.to_csv('init_args.csv',sep=';',index=False)
    print('Running')
    runpy.run_path('init.py')
    print('Stored data to data/{}'.format(out_name))
    
def mock(grid = False,St=0.01,v_f=2,delta=0.0001,
         free_iron=True,carbon_grain=True,many=False,
         viscous_heating=False,constant_flux=False):
      
    out_name = 'mock_joanna'
    in_name = 'mock_SAPP_scaling'
    init_args = pd.read_csv('init_args.csv',sep=';')
    
    init_args['sun'] = 0
    init_args['St'] = St
    init_args['dt'] = 0.001
    init_args['r_low'] = 0.5
    init_args['t_low'] = 0.01
    init_args['N_bodies'] = 10000
    init_args['sun'] = 0
    init_args['store_full'] = 0
    init_args['free_iron'] = int(free_iron)
    init_args['carbon_grain'] = int(carbon_grain)
    init_args['evap'] = 1
    init_args['overwrite'] = 1
    init_args['v_f']=v_f
    init_args['delta']=float(delta)
    init_args['viscous_heating']=int(viscous_heating)
    init_args['constant_flux'] = int(constant_flux)
    
    if many:
        
        in_name+='_many'
        out_name+='_many'
        init_args['N_bodies'] = 2*10**3
    
    init_args['input_name'] = in_name
    
    if not grid:
        
        init_args['grid'] = 0
        
    elif grid:
                        
        out_name+='_grid'

        init_args['grid'] = 1
    
    if free_iron:
        
        out_name+='_freeiron'

    if constant_flux:
        
        out_name+='_constantflux'
        
    if not carbon_grain:
        
        out_name +='_nograin'
    
    if v_f != 2:
        
        out_name+='_vf{}'.format(str(v_f))
    
    if delta != 0.0001:

        out_name+='_delta{}'.format(str(int(np.log10(delta)))[1])
               
    if St != 0.01:
        
        out_name+='_St{}'.format(str(np.log10(St))[1])
    
    init_args['output_name'] = out_name
    init_args.to_csv('init_args.csv',sep=';',index=False)
    print('Running')
    runpy.run_path('init.py')
    print('Stored data to data/{}'.format(out_name))

def cats(v_f=2,St=0.01,delta=0.0001,free_iron=True,carbon_grain=True,
         constant_flux = False,viscous_heating = False):
    
    cat_names = ["thin","thick","halo"]

    init_args = pd.read_csv("init_args.csv",delimiter=";")
    init_args["sun"] = 0
    init_args["N_bodies"] = 2000
    init_args['St'] = St
    init_args['dt'] = 0.001
    init_args['r_low'] = 0.5
    init_args['t_low'] = 0.01
    init_args['store_full'] = 0
    init_args['grid'] = 0
    init_args['free_iron'] = int(free_iron)
    init_args['carbon_grain'] = int(carbon_grain)
    init_args['viscous_heating'] = int(viscous_heating)
    init_args['evap'] = 1
    init_args['overwrite'] = 1
    init_args['constant_flux'] = int(constant_flux)
    init_args['v_f']=v_f
    init_args['delta']=float(delta)
    
    name_add=''
    
    if not carbon_grain:
        
        name_add+='_nograin'
    
    if free_iron:
        
        name_add+='_freeiron'
    
    if constant_flux:
        
        name_add+='_constantflux'
    
    if v_f != 2:
        
        name_add+='_vf{}'.format(str(v_f))
    
    if delta != 0.0001:

        name_add+='_delta{}'.format(str(delta).split('.')[1])
        
    if St != 0.01:
        
        name_add+='_St{}'.format(str(np.log10(St))[1])

    for name in cat_names:
        
        out_name = name+name_add
        
        filename = "SAPP_{}_snr_cut_50_input".format(name)
        
        init_args["input_name"] = filename
        init_args["output_name"] = out_name
        
        init_args.to_csv("init_args.csv",index=False,sep=";")
        
        print("Running {}".format(name))
        runpy.run_path('init.py')
        print('Stored data to data/{}'.format(out_name))

def feh_mass(v_f=2,St=0.01,delta=0.0001,free_iron=True):
    
    init_args = pd.read_csv('init_args.csv',sep=';')
    
    init_args['input_name'] = 'mock_feh_mass_many'
    init_args['St'] = St
    init_args['dt'] = 0.001
    init_args['r_low'] = 0.5
    init_args['t_low'] = 0.01
    init_args['N_bodies'] = 2000
    init_args['sun'] = 0
    init_args['store_full'] = 0
    init_args['grid'] = 0
    init_args['free_iron'] = int(free_iron)
    init_args['carbon_grain'] = 1
    init_args['evap'] = 1
    init_args['overwrite'] = 1
    init_args['constant_flux'] = 0
    init_args['viscous_heating'] = 0
    
    init_args['v_f']=v_f
    init_args['delta']=float(delta)
    
    out_name = 'mock_feh_mass_many'
    
    if free_iron:
        
        out_name+='_freeiron'
        
    init_args['output_name'] = out_name
    
    init_args.to_csv('init_args.csv',sep=';',index=False)
    
    print('Running mock feh mass')
    runpy.run_path('init.py')
    
def run_all():
    
    #sun()

    #mock()

    #cats()
    
    run_mean_mass_feh()
    
    feh_mass()
    
    cats(free_iron=True)
    
def run_mean_mass_feh():
    
    mock(many=True)
    mock(v_f=1,many=True)
    mock(v_f=5,many=True)
    mock(delta=0.001,many=True)
    mock(delta=0.00001,many=True)
    #mock(free_iron=True,many=True)

def run_mean_mass_feh_constant():
    
    mock(many=True,constant_flux=True)
    mock(St=0.1,many=True,constant_flux=True)
    mock(St=0.001,many=True,constant_flux=True)
    mock(delta=0.001,many=True,constant_flux=True)
    mock(delta=0.00001,many=True,constant_flux=True)

