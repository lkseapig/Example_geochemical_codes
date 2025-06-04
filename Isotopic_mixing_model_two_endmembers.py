
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Load data
geochem = pd.read_excel('Data/TVZ_WR_data_compilation_22.08.24.xlsx', sheet_name="Data")
#%%

sample_loc = ['PP', 'MP']
samples = geochem[geochem['data_source'].isin(sample_loc)]
samples = samples.loc[(samples['SiO2n'] >= 53)]

# Load basement data for different terranes
basement = pd.read_excel('Data/Copy of TVZ_Metasedimentary_Basement_Compilation.xlsx', sheet_name='Pb_isotopes')

# Filter E-MORB and N-MORB data
kaweka_data = basement[basement['Our Terrane'] == 'Kaweka']
pahau_data = basement[basement['Our Terrane'] == 'Pahau']
waipapa_data = basement[basement['Our Terrane'] == 'Waipapa']

# Calculate the mean for each basement type. 
kaweka_mean = kaweka_data.mean()
pahau_mean = pahau_data.mean()
waipapa_mean = waipapa_data.mean()

# Convert Series to DataFrame with a single row
kaweka_mean_df = pd.DataFrame([kaweka_mean], columns=kaweka_data.columns)
pahau_mean_df = pd.DataFrame([pahau_mean], columns=pahau_data.columns)
waipapa_mean_df = pd.DataFrame([waipapa_mean], columns=waipapa_data.columns)

# Basement and basalt samples used for modelling. 
basalt = kaweka_mean_df

basalt_loc = ['R858']
basement_loc = geochem.loc[geochem['Sample'].isin(basalt_loc)]

# Extract basement data
Ea_basement_6_4 = basement_loc['206Pb/204Pb'].to_numpy()[0]
Ea_basement_7_4 = basement_loc['207Pb/204Pb'].to_numpy()[0]
Ea_basement_8_4 = basement_loc['208Pb/204Pb'].to_numpy()[0]
Ea_basement_7_6 = basement_loc['207Pb/206Pb'].to_numpy()[0]
Ea_basement_8_6 = basement_loc['208Pb/206Pb'].to_numpy()[0]
Ca_basement_6_4 = basement_loc['Pb'].to_numpy()[0]
Ca_basement_7_4 = basement_loc['Pb'].to_numpy()[0]
Ca_basement_8_4 = basement_loc['Pb'].to_numpy()[0]
Ca_basement_7_6 = basement_loc['Pb'].to_numpy()[0]
Ca_basement_8_6 = basement_loc['Pb'].to_numpy()[0]

# Extract basalt data
Em0_basalt_6_4 = basalt['206Pb/204Pb'].to_numpy()[0]
Em0_basalt_7_4 = basalt['207Pb/204Pb'].to_numpy()[0]
Em0_basalt_8_4 = basalt['208Pb/204Pb'].to_numpy()[0]
Em0_basalt_7_6 = basalt['207Pb/206Pb'].to_numpy()[0]
Em0_basalt_8_6 = basalt['208Pb/206Pb'].to_numpy()[0]
Cm0_basalt_6_4 = basalt['Pb'].to_numpy()[0]
Cm0_basalt_7_4 = basalt['Pb'].to_numpy()[0]
Cm0_basalt_8_4 = basalt['Pb'].to_numpy()[0]
Cm0_basalt_7_6 = basalt['Pb'].to_numpy()[0]
Cm0_basalt_8_6 = basalt['Pb'].to_numpy()[0]

# Internal input values
r_values = np.linspace(0, 0.2, 3)
# Modelling Pb isotopes, so both use the same value. 
#The D_sr is in there in case you want to model Sr versus Pb isotopes, just change the value here. 
D_sr = 1.265
D_pb = 1.265

# Define texture and rock type labels and dictionaries
Rotoiti_labels = ['Magma_1', 'Magma_2', 'Group_1', 'Group_2', 'Mixed', 'Unknown', 'Background', 'Group_3', 'Matahi', 'Test']
Rotoiti_dict = dict(Magma_1=0, Magma_2=1, Group_1=2, Group_2=3, Mixed=4, Unknown=5, Background=6, Group_3 = 7, Matahi = 8, Test = 9)
colors = ['#f28e2b', '#f28e2b', '#76b7b2', '#9c755f', '#f28e2b', '#f28e2b', 'black', '#ff9da7', '#f28e2b', '#f28e2b']

rock_labels = ["MP", "PP", "PV", "MV"]
rock_dict = dict(MP=0, PP=1, PV=2, MV=3)
markers = ['o', 'X', 's', 'd']

# Define function to calculate variable Z values
def z_calc(r, D):
    return ((r + D - 1) / (r - 1))

# Define isotope AFC function
def afc_iso(r, Ca, Cm, f, z, Ea, Cm0, Em0):
    Em_func = ((r / (r - 1)) * Ca / z * (1 - (f ** -z)) * Ea + Cm0 * (f ** -z) * Em0) / (((r / (r - 1)) * Ca / z * (1 - (f ** -z)) + Cm0 * f ** -z))
    return Em_func

    # Define isotope AFC function
def afc_test(Ca, Cm, f, z, Ea, Cm0, Em0):
    Em_func = (Ea - Em0) * (1 - (Cm0 / Cm) * f**-z) + Em0
    return Em_func

#%%
def plot_mixing_lines(ax, isotope_pair, samples, r_values, show_legend=True, xlim=None, ylim=None):
    f_values = np.linspace(0, 1, 11)
    
    isotope_mapping = {
        '206Pb/204Pb': '6_4',
        '208Pb/204Pb': '8_4',
        '207Pb/204Pb': '7_4',
        '207Pb/206Pb': '7_6',
        '208Pb/206Pb': '8_6'
    }
    
    Ea_x, Ea_y = globals()[f'Ea_basement_{isotope_mapping[isotope_pair["x"]]}'], globals()[f'Ea_basement_{isotope_mapping[isotope_pair["y"]]}']
    Ca_x, Ca_y = globals()[f'Ca_basement_{isotope_mapping[isotope_pair["x"]]}'], globals()[f'Ca_basement_{isotope_mapping[isotope_pair["y"]]}']
    Cm0_x, Cm0_y = globals()[f'Cm0_basalt_{isotope_mapping[isotope_pair["x"]]}'], globals()[f'Cm0_basalt_{isotope_mapping[isotope_pair["y"]]}']
    Em0_x, Em0_y = globals()[f'Em0_basalt_{isotope_mapping[isotope_pair["x"]]}'], globals()[f'Em0_basalt_{isotope_mapping[isotope_pair["y"]]}']

    
    r_lines = []  # Collect handles for R value lines
    for r_val in r_values:
        Em_x_values = []
        Em_y_values = []
        
        for f in f_values:
            z_x = z_calc(r_val, D_pb)
            z_y = z_calc(r_val, D_sr)
            Em_x_value = afc_iso(r_val, Ca_x, Cm0_x, f, z_x, Ea_x, Cm0_x, Em0_x)
            Em_y_value = afc_iso(r_val, Ca_y, Cm0_y, f, z_y, Ea_y, Cm0_y, Em0_y)
            Em_x_values.append(Em_x_value)
            Em_y_values.append(Em_y_value)
        
        line_handle, = ax.plot(Em_x_values, Em_y_values, label=f'r = {r_val:.2f}', marker='o', markersize = 7, zorder = 10)
        r_lines.append(line_handle)
    
    # Plotting sample points with error bars
    for labels, d in samples.groupby(['Rotoiti_group_new', 'data_source']):
        if labels[1] not in rock_dict:
            continue
        # Specific style for "Background"
        if labels[0] in ['Background']:
            edgecolor = 'black'
            facecolor = 'none'
            error_color = 'black'
            zorder = 1
        else:
            # color = colors[Rotoiti_dict[labels[0]]]
            color = 'none'
            edgecolor = color if labels[1] in ["PP", "PV", "EV", "MV"] else 'black'
            edgecolor = 'black'
            facecolor = 'none' if labels[1] in ["PP", "PV", "EV", "MV"] else color
            error_color = 'black'

        
        # Set the zorder based on the data source priority: MP > PP > PV > EV
        if labels[1] == "MP":
            zorder = 7  # Highest priority
        elif labels[1] == "PP":
            zorder = 6
        elif labels[1] in ["PV", "EV"]:
            zorder = 5
        else:
            zorder = 4  # Default for other categories

        
        
        marker = markers[rock_dict[labels[1]]]
        # zorder = 5  # Adjust zorder if needed

        # Determine if yerr and xerr columns are present
        yerr = d[f'{isotope_pair["y"]} error'] if f'{isotope_pair["y"]} error' in d.columns else None
        xerr = d[f'{isotope_pair["x"]} error'] if f'{isotope_pair["x"]} error' in d.columns and not d[f'{isotope_pair["x"]} error'].isnull().all() else None

        # Plot the data points with error bars if errors are available
        if xerr is not None and yerr is not None:
            ax.errorbar(d[isotope_pair["x"]], d[isotope_pair["y"]], yerr=yerr, xerr=xerr, fmt=marker, markersize=8,
                        markerfacecolor=facecolor, markeredgecolor=edgecolor,
                        capsize=2, label=labels[0], ecolor=error_color, elinewidth=0.5, zorder=zorder)
        elif xerr is not None:
            ax.errorbar(d[isotope_pair["x"]], d[isotope_pair["y"]], xerr=xerr, fmt=marker, markersize=8,
                        markerfacecolor=facecolor, markeredgecolor=edgecolor, capsize=2, label=labels[0], ecolor=error_color, elinewidth=0.5, zorder=zorder)
        elif yerr is not None:
            ax.errorbar(d[isotope_pair["x"]], d[isotope_pair["y"]], yerr=yerr, fmt=marker, markersize=8,
                        markerfacecolor=facecolor, markeredgecolor=edgecolor, capsize=2, label=labels[0], ecolor=error_color, elinewidth=0.5, zorder=zorder)
        else:
            ax.plot(d[isotope_pair["x"]], d[isotope_pair["y"]], linestyle='', marker=marker, markersize=8,
                    markerfacecolor=facecolor, markeredgecolor=edgecolor, label=labels[0], zorder=zorder)
                
        # Plot basement data
        handles_basement = []
        
        # Plotting the mean value for each greywacke terrane
        for terrane, loc, color, marker in zip(['Kaweka', 'Waipapa', 'Pahau'], [kaweka_mean_df, waipapa_mean_df, pahau_mean_df], ['green', 'blue', 'purple'], ['s', 's', 's']):
            scatter_handle = plt.Line2D([0], [0], color=color, marker=marker, linestyle='None', markersize=10)
            handles_basement.append((scatter_handle, f'Basement {terrane}'))
            ax.scatter(loc[isotope_pair['x']], loc[isotope_pair['y']], color=color, label=f'Basement {terrane}', s=100, marker=marker)
        
        # Plotting the all the data for each greywacke terrane
        for terrane, loc, color, marker in zip(['Kaweka', 'Waipapa', 'Pahau'], [kaweka_data, waipapa_data, pahau_data], ['green', 'blue', 'purple'], ['o', 's', '^']):
            scatter_handle = plt.Line2D([0], [0], color=color, marker=marker, linestyle='None', markersize=10)
            handles_basement.append((scatter_handle, f'Basement {terrane}'))
            # ax.scatter(loc[isotope_pair['x']], loc[isotope_pair['y']], color=color, label=f'Basement {terrane}', s=100, marker=marker)
            # # Plot KDE for each terrane
            
        
            sns.kdeplot(
                x=loc[isotope_pair['x']], 
                y=loc[isotope_pair['y']], 
                    fill=True, 
                    color=color, 
                    alpha=0.01,  # Adjust transparency as needed
                    thresh = 0.05,
                    levels=2,  # Number of contour levels
                    bw_adjust = 0.5,
                    ax=ax, 
                    zorder = 0
                    )
                    
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    # Set labels and limits
    ax.set_xlabel(isotope_pair["x"], fontsize = 14)
    ax.set_ylabel(isotope_pair["y"])
    ax.grid(False)
    ax.set_facecolor('white')

    if show_legend:       
        handles = r_lines
        labels =  [f'r = {r_val:.2f}' for r_val in r_values]

        axs[0,0].legend(handles, labels, loc='upper left')


# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Define isotope pairs and axis limits
isotope_pairs = [
    {"x": "206Pb/204Pb", "y": "208Pb/204Pb"},
    {"x": "207Pb/204Pb", "y": "208Pb/204Pb"},
    {"x": "206Pb/204Pb", "y": "207Pb/204Pb"}, 
    {"x": "207Pb/206Pb", "y": "208Pb/206Pb"}
]


axis_limits = {
    0: {'xlim': (18.65, 19.0), 'ylim': (38.54, 38.9)},
    1: {'xlim': (15.60, 15.66), 'ylim': (38.54, 38.9)},
    2: {'xlim': (18.65, 19.0), 'ylim': (15.6, 15.66)}, 
    3: {'xlim': (0.826, 0.834), 'ylim': (2.048, 2.062)},
}


# Define basement data info
basement_info = [
    {'data': kaweka_data, 'color': 'green', 'label': 'Kaweka basement'},
    {'data': waipapa_data, 'color': 'blue', 'label': 'Waipapa basement'},
    {'data': pahau_data, 'color': 'purple', 'label': 'Pahau basement'}
]

# Plot mixing lines and basement data
for i, ax in enumerate(axs.flatten()):
    plot_mixing_lines(ax, isotope_pairs[i], samples, r_values, show_legend=(i == 3), 
                       xlim=axis_limits[i]['xlim'], ylim=axis_limits[i]['ylim'])
    
    
plt.tight_layout()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

