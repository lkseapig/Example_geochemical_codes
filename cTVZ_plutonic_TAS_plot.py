
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import make_interp_spline  # For smooth interpolation
import numpy as np
#%%


# Define the updated coordinates for each line
lines = [
    [(37, 3), (41, 3)],
    [(41, 3), (45, 3)],
    [(41, 0), (41, 7)],
    [(37, 3), (35, 9)],
    [(35, 9), (37, 14)],
    [(37, 14), (52.5, 18)],
    [(52.5, 18), (57, 18)],
    [(57, 18), (63, 16.2)],
    [(63, 16.2), (71.8, 13.5)],
    [(71.8, 13.5), (85.9, 6.8)],
    [(85.9, 6.8), (87.5, 4.7)],
    [(87.5, 4.7), (77.3, 0)],
    [(69, 8), (77.3, 0)],
    [(52, 5), (69, 8)],
    [(69, 8), (71.8, 13.5)],
    [(69, 8), (77.3, 0)],
    [(45, 5), (61, 13.5)],
    [(61, 13.5), (63, 16.2)],
    [(52.5, 14), (52.5, 18)],
    [(41, 7), (52.5, 14)],
    [(63, 7), (63, 0)],
    [(57, 5.9), (57, 0)],
    [(52, 5), (52, 0)],
    [(45, 5), (45, 0)],
    [(52.5, 14), (57.6, 11.7)],
    [(48.4, 11.5), (53, 9.3)],
    [(45, 9.4), (49.4, 7.3)],
    [(49.4, 7.3), (52, 5)],
    [(37, 3), (45, 3)],
    [(45, 5), (52, 5)],
    [(53, 9.3), (57, 5.9)],
    [(57.6, 11.7), (63, 7)],
    [(61, 8.6), (71.8, 13.5)],
]



# Define unique labels for each field
labels = [
    'Peridot-\ngabbro',
    'Foidgabbro',
    'Gabbroic\nDiorite',
    'Foidolite',
    'Foid Monzosyenite',
    'Foid\nMonzodiorite',
    'Foid Syenite',
    'Syneite',
    'Quartz\nMonzonite',
    'Granite',
    'Monzo-\ngabbro',
    'Monzodiorite',
    'Granodiorite',
    'Gabbro',
    'Monzonite',
    'Diorite',

]

# Add labels to the middle of each defined field
label_positions = [
    (43, 0.8),    # Midpoint for (37,3) to (41,3)
    (44.5, 7),    # Midpoint for (41,3) to (45,3)
    (54, 2),  # Midpoint for (41,0) to (41,7)
    (42, 11),    # Midpoint for (37,3) to (35,9)
    (52.5, 11.5), # Midpoint for (35,9) to (37,14)
    (49, 9.5),   # Midpoint for (37,14) to (52.5,18)
    (58, 15),   # Midpoint for (52.5,18) to (57,18)
    (64, 12), # Midpoint for (57,18) to (63,16.2)
    (66, 8.5), # Midpoint for (63,16.2) to (71.8,13.5)
    (78.9, 3), # Midpoint for (71.8,13.5) to (85.9,6.8)
    (49, 6), # Midpoint for (85.9,6.8) to (87.5,4.7)
    (53, 6.5),# Midpoint for (87.5,4.7) to (77.3,0)
    (68, 4),   # Midpoint for (69,8) to (77.3,0)
    (48, 3), # Midpoint for (45,2) to (52,5)
    (58, 8), # Midpoint for (52,5) to (69,8)
    (60, 3),# Midpoint for (69,8) to (71.8,13.5)

]

#%% LOAD DATA AND SET PLOTTING CUSTOMISATION
geochem = pd.read_excel('Data/TVZ_WR_data_compilation_22.08.24.xlsx', sheet_name = 'Data')
#%%
uncert = pd.read_excel('Data/TVZ_WR_data_compilation_22.08.24.xlsx', sheet_name="Uncert.")
#%%
# Filter the data
V_p= V= geochem.loc[(geochem['SiO2'] >= 52) & (geochem['data_source no.'] <= 4) & (geochem['data_source'] != 'MV')]
P = geochem.loc[(geochem['SiO2'] >= 52) & (geochem['data_source no.'] <= 2) & (geochem['data_source'] != 'MV')]
V= geochem.loc[(geochem['SiO2'] >= 52) & (geochem['data_source no.'] >2) & (geochem['data_source no.'] <= 4)]
# Drop rows where 'granitoid age' is either NaN or 'Rotoiti'
V = V[V['granitoid age'].notna() & (V['granitoid age'] != 'Rotoiti')]
major_refs = V['Reference'].unique()
formatted_references = "\n".join(major_refs)

#%%

age_labels = ["Old", "Young", "Rotoiti", 'Modern']
age_dict = dict(Old=0, Young=1, Rotoiti=2, Modern=3)
colors = ['#59a14f', '#4e79a7', '#bab0ac', '#b07aa1']  # Tableau 10 colors

rock_labels = ["MP", "PP", "PV"]
rock_dict = dict(MP=0, PP=1, PV=2)
markers = ['o', 'X', 's']

#%%
oxide_uncert = {
    'SiO2n': uncert['SiO2n'].values[0],
    'Na2O+K2On': uncert['Na2O+K2On'].values[0],
    }

x_oxides = ['SiO2n']
y_oxides = ['Na2O+K2On']

#%%

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10.8, 8))

# Plot each line segment
for line in lines:
    x_vals, y_vals = zip(*line)
    ax.plot(x_vals, y_vals, 'k-', lw=1.5)

# Add labels to the middle of each defined area
for pos, label in zip(label_positions, labels):
    ax.text(pos[0], pos[1], label, fontsize=10, ha='center', va='center')

# Define color mapping
color_mapping = dict(zip(age_labels, colors))

# Separate the data into two subsets based on SiO2 content
P_high_SiO2 = P[P['SiO2n'] > 63]  # SiO2 > 62
P_low_SiO2 = P[P['SiO2n'] <= 63] # SiO2 ≤ 62

# Define the plotting order
plot_order = ["Rotoiti", "Modern", "Old", "Young"]

for label in plot_order:
    for data_label, d in P.groupby(['granitoid age', 'data_source']):
        if data_label[0] == label:
            for subset, subset_label in zip(
                [P_high_SiO2, P_low_SiO2],
                ['> 62 wt.% SiO2', '≤ 62 wt.% SiO2']
            ):
                # Subset specific data
                d_subset = subset[
                    (subset['granitoid age'] == data_label[0]) & 
                    (subset['data_source'] == data_label[1])
                ]

                if d_subset.empty:  # Skip if no data in the subset
                    continue

                # Define marker styles and colors based on conditions
                if data_label[0] == 'Rotoiti':
                    if data_label[1] == 'MP':  # Rotoiti with MP data_source
                        facecolor = '#bab0ac'  # Light gray face color
                        edgecolor = '#bab0ac'  # Light gray edge color
                        zorder = 2            # Highest zorder for MP group
                        alpha = 0.5
                        
                    elif data_label[1] == 'PP':  # Rotoiti with PP data_source
                        facecolor = 'none'     # No fill
                        edgecolor = '#bab0ac'  # Light gray edge color
                        zorder = 2             # Lower zorder
                        alpha = 0.5
                    else:
                        continue  # Skip other data_source for Rotoiti
                elif data_label[1] == 'MP':
                    facecolor = colors[age_dict[data_label[0]]]  # Color based on age
                    edgecolor = 'black'        # Black edge color
                    zorder = 11   
                    alpha = 1# Highest zorder for MP group
                else:
                    facecolor = 'none'         # No fill
                    edgecolor = colors[age_dict[data_label[0]]]  # Color based on age
                    zorder = 5
                    alpha = 1# Lower zorder for these groups

                # Determine marker style based on subset label
                marker = 'o' if subset_label == '> 62 wt.% SiO2' else 'X'

                # Adjust linewidth based on whether facecolor is 'none'
                edgewidth = 1 if facecolor != 'none' else 1.5

                # Scatter plot for the current subset
                ax.scatter(
                    d_subset[x_oxides], d_subset[y_oxides],
                    label=f"{data_label[0]} ({subset_label})",
                    edgecolor=edgecolor,        # Edge color
                    facecolor=facecolor,        # Fill color
                    marker=marker,              # Marker style
                    alpha=alpha,                  # Transparency
                    s=100,                      # Marker size
                    linewidths=edgewidth,       # Edge width
                    zorder=zorder               # Layer order
                    
                )

#Plot KDE for V_P data points
for label in plot_order:
    for data_label, d in V.groupby(['granitoid age', 'data_source']):
        if data_label[0] == label:
            # Set the color for the KDE plot based on the age group
            color = colors[age_dict[data_label[0]]]
            
            # Plot the KDE for V_P data using 2 levels and with filled contours
            sns.kdeplot(x=d[x_oxides[0]], y=d[y_oxides[0]], 
                        fill=False, color=color, levels=2, thresh=0.1, 
                        alpha=0.5, ax=ax, zorder=1, bw_adjust=0.75, linewidths = 2.5)
            
    # #   # Plot scatter points for V data points
    # for label in plot_order:
    #     for data_label, d in V.groupby(['granitoid age', 'data_source']):
    #         if data_label[0] == label:
    #               # Set the color for the scatter plot based on the age group
    #               color = colors[age_dict[data_label[0]]]
    #               marker = markers[rock_dict[data_label[1]]]
                  
    #               # Plot the scatter points
    #               ax.scatter(d[x_oxides[0]], d[y_oxides[0]], 
    #                           color=color,
    #                           facecolor = 'none',
    #                           alpha = 0.1,
    #                           marker = marker,
    #                           s=100,  # Size of the scatter points
    #                           edgecolor=color,  # Optional: set edge color for visibility
    #                           zorder=1)

# Error bar location and plot
x_loc = 85  # SiO2 value near the center of the plot
y_loc = 11  # Na2O+K2O value within the axis limits

ax.text(x_loc, y_loc -0.7, '2 s.d.', color='black', fontsize=10, ha='center', va='bottom')

ax.errorbar(x=x_loc, y=y_loc, xerr=oxide_uncert['SiO2n'], yerr=oxide_uncert['Na2O+K2On'], 
             ecolor='black', capsize=2, elinewidth=0.8, zorder=4)

# Add the curved line (smooth interpolation)
x_coords = [39, 45, 53, 61, 75]
y_coords = [0, 2.8, 5.5, 8, 12]

# Interpolating the points for smooth curve
x_smooth = np.linspace(min(x_coords), max(x_coords), 500)
spl = make_interp_spline(x_coords, y_coords, k=3)  # Cubic spline interpolation
y_smooth = spl(x_smooth)

# Plot the curved line
ax.plot(x_smooth, y_smooth, 'k--', lw=1.5, label='Curved Line')


# Create custom legend handles for different age groups (color)
age_legend = [
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[age_dict['Old']], markeredgecolor = 'k', markersize=10, label='Old'),
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[age_dict['Young']], markeredgecolor = 'k', markersize=10, label='Young'),
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[age_dict['Modern']], markeredgecolor = 'k', markersize=10, label='Modern'),
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#bab0ac', markersize=10, markeredgecolor = 'none',alpha = 0.5, label='Rotoiti')
]

# # Create custom legend handles for rock types (black edges and different markers)
# rock_legend = [
#     plt.Line2D([0], [0], marker=markers[rock_dict['MP']], color='none', markerfacecolor='k', markeredgecolor = 'k', markersize=10, label='MP'),
#     plt.Line2D([0], [0], marker=markers[rock_dict['PP']], color='none', markerfacecolor='none',markeredgecolor = 'k',  markersize=10, label='PP'),
# ]

# # Combine the two legends
# legend_handles = age_legend + rock_legend
# legend_labels = ['Old-hosted', 'Young-hosted', 'Modern-hosted', 'Rotoiti-hosted', 'Plutonics (this study)', 'Published plutonics']

# # Add the combined legend to the plot
# ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize=12)

# Custom legend handles for age groups, categorized by SiO2 content
age_legend_siO2 = [
    # < 63 wt. % SiO₂ group (cross markers)
    plt.Line2D([0], [0], marker='X', color='none', markerfacecolor=colors[age_dict['Old']], markeredgecolor='k', markersize=10, label='Old-hosted'),
    plt.Line2D([0], [0], marker='X', color='none', markerfacecolor=colors[age_dict['Young']], markeredgecolor='k', markersize=10, label='Young-hosted'),
    plt.Line2D([0], [0], marker='X', color='none', markerfacecolor=colors[age_dict['Modern']], markeredgecolor='k', markersize=10, label='Modern-hosted'),

    # > 63 wt. % SiO₂ group (circle markers)
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[age_dict['Old']], markeredgecolor='k', markersize=10, label='Old-hosted'),
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[age_dict['Young']], markeredgecolor='k', markersize=10, label='Young-hosted'),
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[age_dict['Modern']], markeredgecolor='k', markersize=10, label='Modern-hosted'),
]

# Custom legend handles for rock types
rock_legend = [
    plt.Line2D([0], [0], marker=markers[rock_dict['MP']], color='none', markerfacecolor='k', markeredgecolor='k', markersize=10, label='Plutonics (This Study)'),
    plt.Line2D([0], [0], marker=markers[rock_dict['MP']], color='none', markerfacecolor='none', markeredgecolor='k', markersize=10, label='Published Plutonic'),
]

# Create grouped legend with SiO₂ categories as headings
legend_elements = [
    # < 63 wt. % SiO₂ heading
    plt.Line2D([0], [0], linestyle='None', label='< 63 wt. % SiO₂'),
    *age_legend_siO2[:3],  # Cross markers

    # > 63 wt. % SiO₂ heading
    plt.Line2D([0], [0], linestyle='None', label='> 63 wt. % SiO₂'),
    *age_legend_siO2[3:],  # Circle markers

    # Rock types
    *rock_legend
]

# Add the legend to the plot
ax.legend(handles=legend_elements, loc='upper right')


ax.set_xlabel(r'$SiO_2$ (wt. %)', fontsize=14)
ax.set_ylabel(r'$Na_2O + K_2O$ (wt. %)', fontsize=14)

# Set axis limits
ax.set_xlim(35, 90)
ax.set_ylim(0, 19)

# Show the plot
plt.tight_layout()
plt.show()

# This will replace anything with the same name without warning
# Save the figure as a PNG file with DPI of 300
# plt.savefig('Final_figures\cTVZ_granitoid_plutonics_TAS.png', dpi=300, bbox_inches='tight')
# Set axis labels for a typical TAS diagram

