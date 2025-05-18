import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Read data
geochem = pd.read_excel('Data/TVZ_WR_data_compilation_22.08.24.xlsx', sheet_name='Data')
#%%
# # Filter the data - you can delete this if it is only the data you want to plot in youe excel/csv file
Seelig = geochem.loc[(geochem['SiO2'] >= 54.5) & (geochem['data_source no.'] <= 1)]
Seelig = Seelig.dropna(subset=['Age_REE'])
Seelig = Seelig[Seelig['Sr'].notna()]

#%%
# Read normalization values from the Excel file
norm = pd.read_excel('Data/Spider_diagram_normalisation_values.xlsx', index_col=0)  # Assuming the first column contains row names

# Specify the name of the row for normalization - change to whatever you need based on the normalisation sheet
row_name = "Mcdonough_and_sun_1989"  # Change this to the desired row name
# row_name = "D&P_15"

# Check if the specified row name exists in the DataFrame
if row_name not in norm.index:
    print(f"Row '{row_name}' not found in the normalization values DataFrame.")
else:
    # Select the specified row for normalization
    norm_row = norm.loc[row_name]

    # Make a copy of the DataFrame to keep the original intact
    silica_loc_norm = Seelig.copy()

    # Normalize values in specified columns
    for column in norm.columns:
        if column in silica_loc_norm.columns:
            # Check for zero values in the normalization row
            if norm_row[column] == 0:
                print(f"Warning: Division by zero for column '{column}'. Skipping normalization.")
                continue  # Skip normalization for this column
            # Check for missing values in the normalization row
            if pd.isnull(norm_row[column]):
                print(f"Warning: Missing value in column '{column}' of the normalization row. Skipping normalization.")
                continue  # Skip normalization for this column
            # Perform normalization
            silica_loc_norm[column] = silica_loc_norm[column] / norm_row[column]

#%%
# Define textural groupings using a dictionary comprehension
# Where Age_REE is a column in my excel sheet and 'Old', 'Young' etc are the values under that heading. 
textures = {
    # 'Old': silica_loc_norm.loc[silica_loc_norm['Age_REE'] == 'Old'],
    'Young': silica_loc_norm.loc[silica_loc_norm['Age_REE'] == 'Young'],
    'Modern': silica_loc_norm.loc[silica_loc_norm['Age_REE'] == 'Modern'],
    # 'Rotoiti': silica_loc_norm.loc[silica_loc_norm['Age_REE'] == 'Rotoiti'],
    # 'Group_1': silica_loc_norm.loc[silica_loc_norm['Age_REE'] == 'Group_1'],
    # 'Group_2': silica_loc_norm.loc[silica_loc_norm['Age_REE'] == 'Group_2'],
    # 'FRT3/3': silica_loc_norm.loc[silica_loc_norm['Age_REE'] == 'FRT3/3']
}
#%%
# Define texture labels and colors
age_labels = ["Old", "Young", "Modern", "Rotoiti"]#, "Group_2", "FRT3/3"]
colors = ['#59a14f', '#4e79a7', '#b07aa1', 'gainsboro']#, 'turquoise', 'black']
# Choosing foreground/background order i.e., 3 is highest, so would plot above everything else. 
zorders = {
    'Rotoiti':1,
    'Young': 4,
    'Modern': 2,
    'Old': 3,
}

# For all elements (i.e., multi_element plot) - you can change this here and it will only plot what you specify here. 
Rollinson_21 = ['Cs', 'Rb', 'Ba', 'Th', 'U', 'K2On', 'Nb', 'Ta', 'La', 'Ce', 'Pb', 'Pr',
               'Sr', 'Nd', 'Zr', 'Hf', 'Sm', 'Eu', 'TiO2', 'Gd', 'Tb', 'Dy', 'Ho', 'Y', 'Er', 'Yb', 'Lu']

REE = ['La(N)', 'Ce(N)', 'Pr(N)', 'Nd(N)', 'Sm(N)', 'Eu(N)', 'Gd(N)', 'Tb(N)', 'Dy(N)', 'Ho(N)', 'Er(N)', 'Tm(N)', 'Yb(N)', 'Lu(N)']
REE_2 = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']

#%%
# Separate the data into two subsets based on SiO2 content
P_high_SiO2 = silica_loc_norm[silica_loc_norm['SiO2n'] > 63]  # SiO2 > 62
P_low_SiO2 = silica_loc_norm[silica_loc_norm['SiO2n'] <= 63] # SiO2 ≤ 62

#%%

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting each textural group
for texture, group in textures.items():
    color_index = age_labels.index(texture)
    data_to_plot = group[REE_2]
    
    # Loop through each row of the data and plot
    for idx, row in data_to_plot.iterrows():
        # Convert to numeric and filter finite data
        row = pd.to_numeric(row, errors='coerce')  # Force conversion to numeric, non-numeric becomes NaN
        finite_mask = np.isfinite(row)  # Now this works because we have a pure numeric row
        
        # Use finite data to create x and y values
        x_values = np.arange(len(REE_2))[finite_mask]
        y_values = row[finite_mask].values  # Get only finite y-values
        
        # Plot the line for each row
        ax.plot(x_values, y_values, color=colors[color_index], alpha=0.5, zorder=zorders[texture])

        # Determine zorder for markers based on textural group
        marker_zorder = 3  # Default zorder for markers
        if texture == 'Young':  # Increase zorder for the 'Young' group
            marker_zorder = 5  # Higher zorder for 'Young'

        # Check if the current row corresponds to high SiO2 (SiO2 > 62) or low SiO2 (SiO2 <= 62)
        if idx in P_high_SiO2.index:  # This will check if the row index belongs to high SiO2
            ax.scatter(x_values, y_values, color=colors[color_index], marker='o', zorder=marker_zorder)  # Circle for high SiO2
        elif idx in P_low_SiO2.index:  # This will check if the row index belongs to low SiO2
            ax.scatter(x_values, y_values, color=colors[color_index], marker='x', zorder=marker_zorder)  # X for low SiO2

# Set x-axis ticks and labels
plt.xticks(range(len(REE_2)), REE_2, fontsize=12)
plt.yticks(fontsize=12)

# Set y-axis limits and scale
ax.set_ylim(0.01, 10000)
# ax.set_ylim(1, 1000)
plt.yscale('log')

# Set y-axis tick label format to non-scientific
ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

# Add horizontal dashed lines at tick positions
for tick in ax.yaxis.get_ticklocs():  # Loop through y-axis tick positions
    ax.axhline(y=tick, color='lightgrey', linewidth=0.5)

# Rotate horizontal axis labels
plt.xticks(rotation=0)

# Add a legend
# plt.legend(loc='upper right', fontsize=12)

# Add labels and title
ax.set_ylabel("CI normalised abundances", fontsize=14)

# Show plot
plt.show()





