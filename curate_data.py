# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:00:28 2022

@author: Jesper
"""
import pandas as pd
from astropy.io import fits
import numpy as np
import sys
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import func
sys.exit()
#%% Create csv file from GALAH fits data
columns = ["sobject_id","teff","e_teff","logg","e_logg",
           "log_lum_bstep","e_log_lum_bstep","m_act_bstep","e_m_act_bstep","age_bstep","e_age_bstep",
           "fe_h","e_fe_h","alpha_fe","e_alpha_fe",
           "C_fe","e_C_fe","Si_fe","e_Si_fe","O_fe","e_O_fe","Mg_fe","e_Mg_fe"]

with fits.open("GALAH_DR3_main_allstar_v2.fits") as file_full:
    
    with fits.open("GALAH_DR3_VAC_ages_v2.fits") as file_param:
        
        
        full_data = file_full[1].data
        full_names = full_data.names
    
        gen_filt = (full_data["flag_fe_h"] == 0) & (full_data["flag_sp"] == 0) & (full_data["snr_c3_iraf"] > 30)
    
        spec_filt = (full_data["flag_C_fe"] == 0) & (full_data["flag_Si_fe"] == 0) & (full_data["flag_Mg_fe"]==0) & (full_data["flag_O_fe"]==0) & (full_data["flag_alpha_fe"]==0)
    
        full_filt = gen_filt & spec_filt
    
        full_data_clean = full_data[full_filt]
        
        param_data = file_param[1].data
        param_names = param_data.names

        idx_in_param = np.array([i in full_data_clean["sobject_id"] for i in param_data["sobject_id"]])
        
        param_data_clean = param_data[idx_in_param]
        
nrows = full_data_clean.shape[0]
ncols = len(columns)

pd_data = np.zeros((nrows,ncols))

for i_col,column in enumerate(columns):
    
    if column in full_names:
        
        pd_data[:,i_col] = full_data_clean[column]
    
    elif column in param_names:
        
        pd_data[:,i_col] = param_data_clean[column]
    
    else:
        
        print("Column not in any data",column)

galah_df = pd.DataFrame(pd_data,columns = columns)
galah_df.to_csv("galah_cleaned.csv",sep=",",index=False)

#%% Rename galah columns

galah_data = pd.read_csv("galah_cleaned.csv")

columns = ["sobject_id","teff","e_teff","logg","e_logg",
           "log_lum_bstep","e_log_lum_bstep","m_act_bstep","e_m_act_bstep","age_bstep","e_age_bstep",
           "fe_h","e_fe_h","alpha_fe","e_alpha_fe",
           "C_fe","e_C_fe","Si_fe","e_Si_fe","O_fe","e_O_fe","Mg_fe","e_Mg_fe"]

galah_new_columns = ["sobject_id","teff","e_teff","logg","e_logg",
           "loglum","e_loglum","mass","e_mass","age","e_age",
           "Fe/H","e_Fe/H","alpha/Fe","e_alpha/Fe",
           "C/Fe","e_C/Fe","Si/Fe","e_Si/Fe","O/Fe","e_O/Fe","Mg/Fe","e_Mg/Fe"]

galah_data.columns = galah_new_columns

galah_data.to_csv("galah_cleaned.csv",sep=",",index=False)


#%% Separate GALAH data into high-alpha/low-alpha

galah_data = pd.read_csv("galah_cleaned.csv")

#From Gandhi & Ness (2019)

high_alpha = galah_data["alpha/Fe"] > -0.08*galah_data["Fe/H"]+0.14

seq_data = np.empty(len(galah_data),dtype=str)

seq_data[high_alpha] = "h"
seq_data[~high_alpha] = "l"

galah_data["alpha_seq"] = seq_data

galah_data.to_csv("galah_cleaned.csv",sep=",",index=False)

#%% Remove nan data from GALAH data

galah_data = pd.read_csv("galah_cleaned.csv")
galah_data.dropna(inplace=True)
galah_data.to_csv("galah_cleaned.csv",sep=",",index=False)


#%% Rescale data based on GALAH and create input file

elements = ["C","Mg","Si","O","Fe","S","He"]

galah_data = pd.read_csv("galah_cleaned.csv")
solar_abun = pd.read_csv("solar_abun.csv")
el_data = pd.read_csv("element_data.csv",delimiter = ";")

galah_in = np.zeros((len(galah_data),len(elements)))

n_fe_h = 10**(galah_data["Fe/H"].values)*solar_abun["Fe/H"].loc[0]

n_x_fe_sun = solar_abun.loc[0]/solar_abun["Fe/H"].loc[0]

columns = ["name"]

for i_el,element in enumerate(elements):
    
    index = element+"/Fe"
    
    columns.append(element+"/H")
    
    if element == "S":
        
        galah_in[:,i_el] = (10**galah_data["Si/Fe"].values)*n_x_fe_sun["S/H"]*n_fe_h
    
    elif element == "He":
        
        continue
    
    elif element != "Fe":
    
        galah_in[:,i_el] = (10**galah_data[index].values)*n_x_fe_sun[element+"/H"]*n_fe_h
    
    else:
        
        galah_in[:,i_el] = n_fe_h
    

columns+=["mass","loglum","teff"]
    
N_star = len(galah_data)

galah_in = np.insert(galah_in,0,galah_data["sobject_id"],axis = 1)
galah_in = np.append(galah_in,galah_data["mass"].values.reshape(N_star,1),axis = 1)
galah_in = np.append(galah_in,galah_data["loglum"].values.reshape(N_star,1),axis = 1)
galah_in = np.append(galah_in,galah_data["teff"].values.reshape(N_star,1),axis = 1)


galah_in_df = pd.DataFrame(galah_in,columns = columns)


Y = (-0.0564*galah_data["Fe/H"].values+0.24)

n_z_h = 0
for i,el in enumerate(elements):
    if el == "He":
        continue
    idx = np.where(el_data["Element"]==el)[0][0]
    
    n_z_h+=el_data["Mass"].values[idx]*galah_in_df[el+"/H"].values

n_h_tot = (1-Y)/(1+n_z_h)

galah_in_df["He/H"] = Y/4/n_h_tot

galah_in_df.to_csv("galah_in.csv",sep=",",index=False)

#%% Take a subset of GALAH data 

N_star = 200

galah = pd.read_csv("galah_cleaned.csv")
galah_in = pd.read_csv("galah_in.csv")

high = galah_in.loc[galah["alpha_seq"] == "h"]
low = galah_in.loc[galah["alpha_seq"] == "l"]

idx_high = np.random.choice(len(high),N_star,replace=False)
idx_low = np.random.choice(len(low),N_star,replace=False)

high_in = high.iloc[idx_high]
low_in = low.iloc[idx_low]

high_in["name"] = np.array(["h"+str(int(i)) for i in high_in["name"].values])
low_in["name"] = np.array(["l"+str(int(i)) for i in low_in["name"].values])


full_in = pd.concat([high_in,low_in])

full_in.to_csv("subset_big_in.csv",index = False,float_format="%.6f")

#%% Create Mock data where abundances scale as Mg based on SAPP data from Matt

data = pd.read_csv("SAPP_total_data_Jesper_final_snr_cut_50.csv")
solar_abun = pd.read_csv("solar_abun.csv")
el_data=pd.read_csv("element_data.csv",delimiter=";")

mgfe = data["mgfe_reso_bay"].values
mgfe_err = data["mgfe_err_reso_bay"].values

feh = data["feh_reso_bay"].values
feh_err = data["feh_err_reso_bay"].values

labels = ["C/Fe","Si/Fe","Mg/Fe","O/Fe"]

def f(x,a,b):
    
    return a*x+b

N_grid = 30
N_star = N_grid**2
new_feh = np.linspace(-0.5,0.5,N_grid)
mass = np.linspace(0.5,1.4,N_grid)

new_feh,mass = np.meshgrid(new_feh,mass)
new_feh=new_feh.flatten()
mass=mass.flatten()

elements = ["C","Mg","Si","O","Fe","S","He","N"]

mock_in = np.zeros((N_star,len(elements)))

n_fe_h = (10**feh)*solar_abun["Fe/H"].loc[0]
n_x_fe_sun = solar_abun.loc[0]/solar_abun["Fe/H"].loc[0]

columns = []

n_mg_fe = 10**(mgfe)*n_x_fe_sun["Mg/H"]

mgh = np.log10(n_mg_fe*n_fe_h)-np.log10(solar_abun["Mg/H"].loc[0])

param,cov = curve_fit(f,feh,mgh,sigma=mgfe_err)

for i_el,element in enumerate(elements):
    
    k = np.random.normal(param[0],cov[0,0],N_star)
    m = np.random.normal(param[1],cov[1,1],N_star)
    
    #k = param[0]
    #m = param[1]
    
    new_data = f(new_feh,k,m)
    
    index = element+"/Fe"
    
    columns.append(element+"/H")
        
    if element == "He":
        
        continue
    
    elif element != "Fe":
    
        mock_in[:,i_el] = (10**new_data)*solar_abun[element+"/H"].loc[0]
    
    else:
        
        mock_in[:,i_el] = (10**new_feh)*solar_abun["Fe/H"].loc[0]
    
columns+=["mass"]
#mock_in = np.append(mock_in,np.ones((3,1)),axis = 1)
mock_in = np.append(mock_in,mass.reshape(N_star,1),axis=1)
#mock_in = np.append(mock_in,np.ones((3,1)),axis = 1)
mock_df = pd.DataFrame(mock_in,columns = columns)
mock_df["name"] = np.arange(N_star)


Y = (-0.0564*mock_df["Fe/H"].values+0.24)

n_z_h = 0
for i,el in enumerate(elements):
    if el == "He":
        continue
    idx = np.where(el_data["Element"]==el)[0][0]
    
    n_z_h+=el_data["Mass"].values[idx]*mock_df[el+"/H"].values

n_h_tot = (1-Y)/(1+n_z_h)

mock_df["He/H"] = Y/4/n_h_tot

mock_df.to_csv("mock_feh_mass_many.csv",index=False,sep=",")

#%% Remove whitespace from SAPP data

data = pd.read_csv("SAPP_total_data_Jesper_final_snr_cut_50.txt",delimiter=",",dtype=None)
for i,col in enumerate(data.columns):
    
    if i in [0,17,30]:
        
        continue
    
    data[col] = data[col].astype(float)

new_col = [i.strip() for i in data.columns]
new_col[0] = "cname"
data.columns=new_col

data.drop(columns=["r_hi_photogeo.1"],inplace=True)
data.to_csv("SAPP_total_data_Jesper_final_snr_cut_50.csv",index=False)
#%% Remove massive stars

data = pd.read_csv("SAPP_total_data_Jesper_final_snr_cut_50.csv")
data_clean = data.loc[data['mass_wa_bay']<1.4]
data_clean.to_csv("SAPP_total_data_Jesper_final_snr_cut_50_nomassive.csv",index=False)

#%% Create input abundances for all stars and categories

np.random.seed(1)

def f(x,k,m):
    
    return k*x+m

R_sun = 8.18
data = pd.read_csv("SAPP_total_data_Jesper_final_snr_cut_50.csv")
#data = data.loc[data['mass_wa_bay']<1.4]
solar_abun = pd.read_csv("solar_abun.csv")
el_data=pd.read_csv("element_data.csv",delimiter=";")

thin_filt = data["Cat"].values == "Thin_disc"
thick_filt = data["Cat"].values == "Thick_disc"
halo_filt = data["Cat"].values == "Halo"

inner = data["GAL_R"].values/1000 > R_sun
outer = data["GAL_R"].values/1000 < R_sun

mgfe = data["mgfe_reso_bay"].values
mgfe_err = data["mgfe_err_reso_bay"].values

feh = data["feh_reso_bay"].values
feh_err = data["feh_err_reso_bay"].values

categories = (thin_filt,thick_filt,halo_filt)
cat_labels = ["Thin disc","Thick disc","Halo","Inner disc","Outer disc"]
cat_file = ["thin","thick","halo","inner","outer"]
colors = ["red","blue","black","purple","yellow"]
N_cat = len(categories)
line_param = np.zeros((N_cat,2))
line_cov = np.zeros((N_cat,2))
#fig,axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=(8,6))
#axes[-1,-1].axis("off")

for i_cat,cat in enumerate(categories):
    
    irow = i_cat//3
    icol = i_cat%3
    
    feh_cat = feh[cat]
    mgfe_cat = mgfe[cat]
    mgfe_err_cat = mgfe_err[cat]
        
    n_fe_h = 10**(feh_cat)*solar_abun["Fe/H"].loc[0]
    
    n_x_fe_sun = solar_abun.loc[0]/solar_abun["Fe/H"].loc[0]
    
    n_mg_fe = 10**(mgfe_cat)*n_x_fe_sun["Mg/H"]
    
    mgh_cat = np.log10(n_mg_fe*n_fe_h)-np.log10(solar_abun["Mg/H"].loc[0])
        
    mass_cat = data["mass_wa_bay"].values[cat]
    
    N_star = len(feh_cat)

    param,cov = curve_fit(f,feh_cat,mgh_cat,sigma=mgfe_err[cat])
    
    line_param[i_cat] = param
    line_cov[i_cat] = np.diag(cov)
    
    residuals = mgh_cat-f(feh_cat,*param)
        
    #axes[irow,icol].scatter(feh_cat,mgh_cat,c="blue",s=5)
    #feh_plot = np.linspace(np.min(feh_cat),np.max(feh_cat),100)
    #axes[irow,icol].plot(feh_plot,f(feh_plot,*param),color="black")
    #axes[irow,icol].set_title(cat_labels[i_cat])
    
    elements = ["C","N","Mg","Si","O","Fe","S","He"]
    mock_in = np.zeros((N_star,len(elements)))
        
    columns = []
    
    for i_el,element in enumerate(elements):
        
        k = np.random.normal(param[0],cov[0,0],N_star)
        m = np.random.normal(param[1],cov[1,1],N_star)
        
        k = param[0]
        m = param[1]
        #scatter = np.random.normal(0,np.mean(abs(residuals)),N_star)
        #scatter = np.random.normal(0,0.05,N_star)
        scatter = np.zeros(N_star)
        new_data = f(feh_cat,k,m)+scatter
        
        index = element+"/Fe"
        
        columns.append(element+"/H")
            
        if element == "He":
            
            continue
        
        elif element == "Mg":
            
            mock_in[:,i_el] = (10**mgh_cat)*solar_abun["Mg/H"].loc[0]
        
        elif element != "Fe":
        
            mock_in[:,i_el] = (10**new_data)*solar_abun[element+"/H"].loc[0]
        
        else:
            
            mock_in[:,i_el] = n_fe_h
            
    columns+=["mass"]
    mock_in = np.append(mock_in,mass_cat.reshape(N_star,1),axis = 1)
    #star_idx = np.random.choice(N_star,100,replace=False)
    mock_df = pd.DataFrame(mock_in,columns = columns)
    mock_df["name"] = data["cname"].values[cat]
    
    Y = (-0.0564*mock_df["Fe/H"].values+0.24)
    
    n_z_h = 0
    for i,el in enumerate(elements):
        if el == "He":
            continue
        idx = np.where(el_data["Element"]==el)[0][0]
        
        n_z_h+=el_data["Mass"].values[idx]*mock_df[el+"/H"].values
    
    n_h_tot = (1-Y)/(1+n_z_h)
    
    mock_df["He/H"] = Y/4/n_h_tot
    
    #mock_df.to_csv("SAPP_{}_snr_cut_50_input_nomassive.csv".format(cat_file[i_cat]),index=False)

np.savetxt('fit_param.txt', line_param,header='k,m')

#%% Create input abundances with scatter

def f(x,k,m):
    
    return k*x+m

data = pd.read_csv("SAPP_total_data_Jesper_final_snr_cut_50.csv")
solar_abun = pd.read_csv("solar_abun.csv")
el_data = pd.read_csv("element_data.csv",delimiter = ";")

thin_filt = data["Cat"].values == "Thin_disc"
thick_filt = data["Cat"].values == "Thick_disc"
halo_filt = data["Cat"].values == "Halo"

categories = (thin_filt,thick_filt,halo_filt)

feh = data["feh_reso_bay"].values
mgfe = data["mgfe_reso_bay"].values
mgfe_err = data["mgfe_err_reso_bay"].values

N_grid = 10
N_star = N_grid**2
new_feh = np.linspace(-2,0.5,N_grid)
mass = np.linspace(0.5,2.5,N_grid)

new_feh,mass = np.meshgrid(new_feh,mass)
new_feh=new_feh.flatten()
mass=mass.flatten()

N_star = 3

new_feh = np.array([-0.5,0,0.5])
        
n_fe_h = 10**(feh)*solar_abun["Fe/H"].loc[0]

n_x_fe_sun = solar_abun.loc[0]/solar_abun["Fe/H"].loc[0]

n_mg_fe = 10**(mgfe)*n_x_fe_sun["Mg/H"]

mgh = np.log10(n_mg_fe*n_fe_h)-np.log10(solar_abun["Mg/H"].loc[0])
    
param,cov = curve_fit(f,feh,mgh,sigma=mgfe_err)

res = mgh-f(feh,*param)

elements = ["C","N","Mg","Si","O","Fe","S","He"]
mock_in = np.zeros((N_star,len(elements)))
    
columns = []

for i_el,element in enumerate(elements):
    
    k = np.random.normal(param[0],cov[0,0],N_star)
    m = np.random.normal(param[1],cov[1,1],N_star)

    scatter = np.random.normal(0,np.mean(abs(res)),N_star)
        
    new_data = f(new_feh,k,m)+scatter
    
    index = element+"/Fe"
    
    columns.append(element+"/H")
        
    if element == "He":
        
        continue
            
    elif element != "Fe":
    
        mock_in[:,i_el] = (10**new_data)*solar_abun[element+"/H"].loc[0]
    
    else:
        
        mock_in[:,i_el] = func.get_abun(new_feh, solar_abun["Fe/H"].loc[0])
        
columns+=["mass"]
mass = np.ones(N_star)
mock_in = np.append(mock_in,mass.reshape(N_star,1),axis = 1)
mock_df = pd.DataFrame(mock_in,columns = columns)
mock_df["name"] = np.array(["low_fe","solar_fe","high_fe"])
#mock_df["name"] = np.arange(N_star)

Y = (-0.0564*mock_df["Fe/H"].values+0.24)

n_z_h = 0
for i,el in enumerate(elements):
    if el == "He":
        continue
    idx = np.where(el_data["Element"]==el)[0][0]
    
    n_z_h+=el_data["Mass"].values[idx]*mock_df[el+"/H"].values

n_h_tot = (1-Y)/(1+n_z_h)

mock_df["He/H"] = Y/4/n_h_tot

mock_df.to_csv("mock_SAPP_scaling_scatter2.csv",index=False)

#%% Create mock data for three metallicities

data = pd.read_csv("SAPP_total_data_Jesper_final_snr_cut_50_nomassive.csv")

mgfe = data["mgfe_reso_bay"].values

feh = data["feh_reso_bay"].values

new_feh = np.array([-0.5,0,0.5])
mass = np.array([1,1,1])
names = np.array(['low_fe','solar_fe','high_fe'])

mock_data = func.scale_data(feh, mgfe, new_feh,mass=mass,names=names,df=True)

mock_data.to_csv('mock_SAPP_scaling_nomassive.csv',index=False)

#%% Create mock data for wide range of metallicities

data = pd.read_csv("SAPP_total_data_Jesper_final_snr_cut_50_nomassive.csv")

mgfe = data["mgfe_reso_bay"].values

feh = data["feh_reso_bay"].values

new_feh = np.linspace(-0.5,0.5,100)
mass = np.ones(100)
names = np.arange(100)

mock_data = func.scale_data(feh, mgfe, new_feh,mass=mass,names=names,df=True)

mock_data.to_csv('mock_SAPP_scaling_nomassive_many.csv',index=False)

#%% Mock data with varying mass

N_grid = 30

new_feh = np.linspace(-0.5,0.5,N_grid)
mass = np.linspace(0.5,1.4,N_grid)

new_feh,mass = np.meshgrid(new_feh,mass)
new_feh=new_feh.flatten()
mass=mass.flatten()
names = np.arange(N_grid**2)
data = pd.read_csv("SAPP_total_data_Jesper_final_snr_cut_50_nomassive.csv")

mock_data = func.scale_data(feh, mgfe, new_feh,mass=mass,names=names,df=True)

mock_data.to_csv('mock_feh_mass_many_nomassive.csv',index=False)