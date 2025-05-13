
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
geochem = pd.read_excel('Data/K-Trig_EPMA_python_02.09.24.xlsx', sheet_name='Plagioclase')

#%% THIS PLOTS BASED ON CRYSTAL TYPES e.g., cumulate, matrix, etc. 
# Define texture groups and their base colors
texture_groups = {
    'Cumulate': sns.color_palette("tab10")[0],  # Blue
    'Intermediate': sns.color_palette("tab10")[1],  # Orange
    'Basalt': sns.color_palette("tab10")[2],  # Green
    'Scoria': sns.color_palette("tab10")[4],  # Purple
}

# Crystal types to use in the histograms
# crystal_types = ['Cumulate', 'Phenocryst', 'Microcumulate', 'Interstitial', 
#                  'Skeletal', 'Matrix', 'Xenocryst', 'Poikilitic', 'Intergrowth']

crystal_types = ['Cumulate/Phenocryst', 'Microcumulate', 'Interstitial', 
                 'Skeletal', 'Matrix', 'Xenocryst', 'Poikilitic', 'Intergrowth']

# Generate shades for each texture's color based on Crystal_type count
shaded_palettes = {texture: sns.light_palette(color, n_colors=len(crystal_types), reverse=True)
                   for texture, color in texture_groups.items()}

# Set the seaborn theme
sns.set_theme(style='ticks')

# Create subplots
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# Define subplot labels
subplot_labels = ["A)", "B)", "C)", "D)"]

# Define y-axis limits for each subplot
y_limits = {
    'Cumulate': (0, 30),
    'Intermediate': (0, 40),
    'Basalt': (0, 20),
    'Scoria': (0, 30)
}

# Loop over textures to create a subplot for each
for i, (texture, base_color) in enumerate(texture_groups.items()):
    row, col = divmod(i, 2)  # Determine row and column index for the subplot
    
    # Filter data for the specific texture
    texture_data = geochem[geochem['Texture'] == texture]
    
    # Map Crystal_type to shades of the texture color
    texture_palette = {crystal_type: shaded_palettes[texture][j]
                       for j, crystal_type in enumerate(crystal_types)}
    
    # Plot histogram with KDE, using 'Crystal_type' as hue for stacking
    sns.histplot(data=texture_data, x="Anorthite", hue="Crystal_type_hist", 
                 multiple="stack", palette=texture_palette, 
                 edgecolor=".3", linewidth=1, ax=ax[row, col], 
                 binwidth=2, binrange=(60, 95), kde=True)
    
    # Set specific y-axis limits for each texture
    ax[row, col].set_ylim(y_limits[texture])
    
    # Set title and subplot label
    sample_size = len(texture_data)
    ax[row, col].set_title(f"{texture} (n={sample_size})", loc='left', x=0.05, y=0.8, fontsize=14)
    ax[row, col].text(0.1, 0.95, subplot_labels[i], transform=ax[row, col].transAxes,
                      fontsize=14, va='top', ha='right')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

#%% THIS PLOTS ABSED ON CRYSTAL LOCATION e.g., core, rim, etc. 

crystal_locations = ['Core', 'Rim', 'Intermediate']

# Generate shades for each texture's color based on Crystal_type count
shaded_palettes = {texture: sns.light_palette(color, n_colors=len(crystal_locations), reverse=True)
                   for texture, color in texture_groups.items()}

# Set the seaborn theme
sns.set_theme(style='ticks')

# Create subplots
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# Define subplot labels
subplot_labels = ["A)", "B)", "C)", "D)"]

# Define y-axis limits for each subplot
y_limits = {
    'Cumulate': (0, 30),
    'Intermediate': (0, 40),
    'Basalt': (0, 20),
    'Scoria': (0, 30)
}

# Loop over textures to create a subplot for each
for i, (texture, base_color) in enumerate(texture_groups.items()):
    row, col = divmod(i, 2)  # Determine row and column index for the subplot
    
    # Filter data for the specific texture
    texture_data = geochem[geochem['Texture'] == texture]
    
    # Map Crystal_type to shades of the texture color
    texture_palette = {crystal_location: shaded_palettes[texture][j]
                       for j, crystal_location in enumerate(crystal_locations)}
    
    # Plot histogram with KDE, using 'Crystal_type' as hue for stacking
    sns.histplot(data=texture_data, x="Anorthite", hue="Crystal_location_hist", 
                 multiple="stack", palette=texture_palette, 
                 edgecolor=".3", linewidth=1, ax=ax[row, col], 
                 binwidth=2, binrange=(60, 95), kde=True,  kde_kws={'bw_adjust': 0.6})
    
    # Set specific y-axis limits for each texture
    ax[row, col].set_ylim(y_limits[texture])
    
    # Set title and subplot label
    sample_size = len(texture_data)
    ax[row, col].set_title(f"{texture} (n={sample_size})", loc='left', x=0.05, y=0.8, fontsize=14)
    ax[row, col].text(0.1, 0.95, subplot_labels[i], transform=ax[row, col].transAxes,
                      fontsize=14, va='top', ha='right')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
