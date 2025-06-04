import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

## Read all relevant excel sheets. 
comp_all = pd.read_excel('Data/DMM_compositions_python.xlsx', sheet_name = 'Compositions')
kd_all = pd.read_excel('Data/DMM_compositions_python.xlsx', sheet_name = 'Kd values')
Df_c0 = pd.read_excel('Data/REE_Kd_values_python.xlsx', sheet_name = 'FC_starting_comps')
Samples = pd.read_excel('Data/REE_Kd_values_python.xlsx', sheet_name = 'Modelled_samples')
Df_kds_ol = pd.read_excel('Data/REE_Kd_values_python.xlsx', sheet_name = 'Olivine')
Df_kds_cpx = pd.read_excel('Data/REE_Kd_values_python.xlsx', sheet_name = 'Cpx')
Df_kds_plag = pd.read_excel('Data/REE_Kd_values_python.xlsx', sheet_name = 'Plag')

 
#%% MODEL FOR BATCH MELTING
# Calculating trace elements in melt from batch melting - not written for residual solid. 
def batch_melt (c0,d,f):
    cl= c0/(d*(1-f) + f)
    return cl

#%% WRITING A MODEL FOR FRACTIONAL MELTING
# LOOPED MODEL FOR VARIABLE D VALUES. 

# Calculating trace elements in melt from fractional melting
def frac_melt (c0,d,f):
    #cl_c0= (1/d)*((1-f)**(1/d-1))
    cl = c0*((1/d)*(1-f)**((1/d)-1))
    return cl
#%% Calculating trace elements in residual solid from fractional melting
# To do proof of validity models, put equations back into form cs/c0 i.e., function above^
def frac_melt_solid (c0,d,f):
    cs= c0*((1-f)**(1/d-1))
    return cs

#%% SELECTING THE MANTLE COMPOSITION FOR THE MANTLE MELTING MODELS 
### Locks the composition to required mantle compositions and Kd values
### Locks the compositions and Kd values to desired columns (REEs at present)
### Need to convert dataframe to an array of floats to run with model

my_f = np.linspace(0, 1, 100)
REE_columns = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']

kd_loc = ['W&M 2005_bulk']
DMM_kd = kd_all.loc[kd_all['Reference'].isin(kd_loc)]
DMM_kd_trim = DMM_kd[REE_columns]
DMM_kd_float = np.array(DMM_kd_trim)
DMM_kd_float = DMM_kd_float.T

c0_loc = ['W&M 2005_2.5_Gpa']
DMM_c0 = comp_all.loc[comp_all['Reference'].isin(c0_loc)]
DMM_c0_trim = DMM_c0[REE_columns]
DMM_c0_float = np.array(DMM_c0_trim)
DMM_c0_float = DMM_c0_float.T

#%% GENERATES MELTED MANTLE COMPOSITIONS WITH VARYING MELT FRACTIONS BASED ON FUNCTION DEFINED IN MODEL
## Plots this up for visual inspection and then outputs an excel sheet with compositions based on F value
# Legend shows starting compositions and partition coefficients

fig, ax  = plt.subplots(figsize = (10,8))

my_cl_DMM_list = []
for FM_loop_c0, FM_loop_d in zip(DMM_c0_float, DMM_kd_float):
    my_cl_DMM = frac_melt(c0=FM_loop_c0, d=FM_loop_d, f=my_f)
    my_cl_DMM_list.append(frac_melt(f=my_f, d=FM_loop_d, c0=FM_loop_c0))
    my_cl_DMM_df = pd.DataFrame(my_cl_DMM_list)
    
    ax.plot(my_f,my_cl_DMM, label='c0 =' + str(FM_loop_c0) + ' , Kd =' + str(FM_loop_d))
    #ax.plot(my_f,my_cl_DMM,label=f'(c0 = {FM_loop_c0:.2f}) (Kd = {FM_loop_d:.2f})')
    ax.legend(loc = 'best')
ax.set_xlabel('Melt Fraction (F)')    
ax.set_ylabel('TE abundance (ppm)')
    
# Transposes dataframe
my_cl_DMM_df_t = my_cl_DMM_df.T
# Renames columns as REE elements
my_cl_DMM_df_t.rename(columns = {0:'La', 1:'Ce', 2:'Pr',
                          3:'Nd', 4:'Sm', 5:'Eu',
                          6:'Gd', 7:'Tb', 8:'Dy',
                          9:'Ho', 10:'Er', 11:'Tm',
                          12:'Yb', 13:'Lu'}, inplace = True)

#Exports generated dataframeto excel file and saves in Data plotting folder
my_cl_DMM_df_t.to_excel("my_cl_DMM_W&M 2005_2.5_Gpa_test.xlsx")


#%% MODEL FOR FRACTIONAL CRYSTALLISATION
# Need to do either batch melting or fractional melting to get the gradient of REEs
#Then put in here to get the abundance of REEs from fractionation. 


ref_ol = 'Rollinson_2021_basalt'
ref_cpx = 'Rollinson_2021_basalt'
ref_plag = 'Rollinson_2021_basalt'

Kd_ol = Df_kds_ol[Df_kds_ol['Reference'] == ref_ol]
Kd_cpx = Df_kds_cpx[Df_kds_cpx['Reference'] == ref_cpx]
Kd_plag = Df_kds_plag[Df_kds_plag['Reference'] == ref_plag]

# Drop the 'Reference' column to only keep the element columns
Kd_ol_elements = Kd_ol.drop(columns=['Reference', 'Comments'])
Kd_cpx_elements = Kd_cpx.drop(columns=['Reference', 'Comments'])
Kd_plag_elements = Kd_plag.drop(columns=['Reference', 'Comments'])

### These are specified here and not in the bulk D calc so the plot can read them later
### in order to add bulk D calc to the REE plot
ol_coeff = 0.5
cpx_coeff = 0.3
plag_coeff = 0.2

# Calculate the bulk partition coefficient
bulk_Kd_elements = (ol_coeff * Kd_ol_elements.values) + (cpx_coeff * Kd_cpx_elements.values) + (plag_coeff *Kd_plag_elements.values)

# Create a new dataframe to store the results
bulk_Kd_df = pd.DataFrame(bulk_Kd_elements, columns=Kd_ol_elements.columns)
bulk_Kd_df_t = bulk_Kd_df.T

print(bulk_Kd_df)

#%% SELECTING THE STARTING COMPOSITION FOR FRACTIONAL CRYSTALLISATION MODEL

c0_loc = ['Taupo_MI_primacalc']
c0_select = Df_c0.loc[Df_c0['Reference'].isin(c0_loc)]

c0_select_trimmed = c0_select.drop(columns=['Reference', 'Comments']) 
c0_trimput_float = c0_select_trimmed.values.flatten().astype(float)

# Changing partition coefficients to floats for model
kd_trimput_float = bulk_Kd_df_t.values.flatten().astype(float)


#%% Defining a function for TEs in melt during fractional crystallisation
def ecfc (c0, x, d):
    cl = c0*((1-x)**(d-1))
    return cl

# Defining a function for TEs in residual solid during fractional crystallisation
# The equations in the white and the rollinson book seem different...?
def fc_rs (c0, x, d):
    cs = c0*(x**(d-1))
    return cs

#%% Plotting trace element abundance versus melt fraction for La through to Lu.
# Order of legend is the same as the order of the 'elements'
# and shows the starting composition used and the partition coefficient used for modelling.

fig, ax = plt.subplots(figsize=(10, 8))

my_x = np.linspace(0, 1, 100)

elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu',
            'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']

elements_array = np.array(elements)

looped_list = []
#for loop_c0, loop_d in zip(my_c0, my_d):
for loop_c0, loop_d in zip(c0_trimput_float, kd_trimput_float):
    my_fc = ecfc(c0 = loop_c0,d = loop_d, x = my_x)
    
    # This one gives you a list with three rows and all the iterations in one value column.
    looped_list.append(ecfc(c0 = loop_c0, d = loop_d, x = my_x)) 
    # Converts to a dataframe
    looped_df = pd.DataFrame(looped_list)#, columns = column_name)
    # Renames rows (index) as REE elements
    looped_df.rename(index = {0:'La', 1:'Ce', 2:'Pr',
                       3:'Nd', 4:'Sm', 5:'Eu',
                       6:'Gd', 7:'Tb', 8:'Dy',
                       9:'Ho', 10:'Er', 11:'Tm',
                       12:'Yb', 13:'Lu'}, inplace = True)
    ax.plot(my_x, my_fc, label=f'(c0 = {loop_c0:.2f}) (Kd = {loop_d:.2f})')
    ax.legend(loc = 'best')
ax.set_xlabel('Melt Fraction (F)')    
ax.set_ylabel('TE abundance (ppm)')
    
# Transposes dataframe
REE_df = looped_df.T

#%% Normalising the REES with CI values from Dauphas and Pourmand 2015

# This creates a tuple for each element 
REE_CI_all = (REE_df['La']/0.2482, REE_df['Ce']/0.6366, REE_df['Pr']/0.0964,
REE_df['Nd']/0.488, REE_df['Sm']/0.1563, REE_df['Eu']/0.06, REE_df['Gd']/0.2102,
REE_df['Tb']/0.0379, REE_df['Dy']/0.2576, REE_df['Ho']/0.0551, REE_df['Er']/0.1654,
REE_df['Tm']/0.0258, REE_df['Yb']/0.1686, REE_df['Lu']/0.0254)

# Converting tuples into dataframe for all CI normalised REEs
REE_CI_df = pd.DataFrame(REE_CI_all)
#Transposing dataframe
REE_CI_df_T = REE_CI_df.T

# Saving output dataframe to excel
#REE_CI_df_T.to_excel("CI_norm_REE_bulkD_test.xlsx")

#%% REE PLOT WITH F (melt fraction) VALUES AT INTERVALS OF 10

REEs = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
fig, ax = plt.subplots(figsize=(12, 8))

sample = Samples[Samples['Reference'] == 'R941A'][REEs].values.flatten()
samples = Samples[Samples['Reference'].isin(['R981', 'R979', 'R532'])][REEs]
sample_R532 = samples.loc[Samples['Reference'] == 'R981'].values.flatten()
sample_R534 = samples.loc[Samples['Reference'] == 'R979'].values.flatten()
sample_R541 = samples.loc[Samples['Reference'] == 'R532'].values.flatten()
c0 = Samples[Samples['Reference'] == 'R986'][REEs].values.flatten()

# Get the Viridis color map
cmap = plt.get_cmap('gray')

# Filter rows at intervals of 10
interval = 10
selected_rows = range(0, len(REE_CI_df_T), interval)

# Number of selected rows
num_rows = len(selected_rows)

# Plot each selected row in the DataFrame with Viridis color map
for i, index in enumerate(selected_rows):
    row = REE_CI_df_T.iloc[index]
    color = cmap(i / (num_rows - 1))  # Normalize index to colormap range
    ax.plot(REEs, row, color=color, label=f'Row {index + 1}')

# Plot sample data
ax.plot(REEs, sample, color='plum', linestyle='--', marker='o', markersize=8, label='R941A')
# # Plot each sample
ax.plot(REEs, sample_R532, color='#f28e2b', linestyle='--', marker='o', markersize=8, label='R981')
ax.plot(REEs, sample_R534, color='#59a14f', linestyle='--', marker='o', markersize=8, label='R979')
ax.plot(REEs, sample_R541, color='#b07aa1', linestyle='--', marker='o', markersize=8, label='R532')

ax.plot(REEs, c0, color='#4e79a7', linestyle='--', marker='o', markersize=8, label='R986')


# Set xticks and labels
ax.set_xticks(range(len(REEs)))
ax.set_xticklabels(REEs, rotation=0, fontsize=14)

# Format the y-axis to show non-scientific notation
ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

# Set axis labels
ax.set_ylabel('CI normalised concentration', fontsize = 14)
ax.set_yscale('log')
ax.tick_params(axis='both', labelsize=14)

# Add label for c0 starting composition
ax.text(0.45, 0.95, f'Starting composition = {c0_loc[0]}', transform=ax.transAxes, fontsize=14, verticalalignment='top', color='black')

bulk_Kd_label = r'Bulk K$_{{d}}$ = {:.2f} * K$_{{d}}^{{ol}}$ + {:.2f} * K$_{{d}}^{{cpx}}$ + {:.2f} * K$_{{d}}^{{plag}}$'.format(ol_coeff, cpx_coeff, plag_coeff)
ax.text(0.45, 0.90, bulk_Kd_label, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

ax.text(0.7, 0.85, f'Kd$_{{ol}}$ = {ref_ol}', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
ax.text(0.7, 0.8, f'Kd$_{{cpx}}$ = {ref_cpx}', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
ax.text(0.7, 0.75, f'Kd$_{{plag}}$ = {ref_plag}', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

# Set axis limits for x-axis and y-axis
ax.set_xlim(-0.5, len(REEs) - 0.5)
ax.set_ylim(1, 1000)  # Adjust y-limits dynamically based on data

# Add a legend and show the plot
ax.legend(loc='lower left')
plt.tight_layout()
plt.show()




