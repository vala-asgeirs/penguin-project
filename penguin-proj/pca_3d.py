import numpy as np
from summary_statistics import df_clean 
from explained_varience import standardized_num_data
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.linalg import svd

U, S, Vt = svd(standardized_num_data, full_matrices=False)

# The principal components are the rows of Vt (transpose of eigenvectors)
# The first 3 principal components can be taken from the first 3 columns of Vt
V = Vt.T  # Transpose to get principal components
X_pca_3d = np.dot(standardized_num_data, V[:, :3])

# Prepare the species data
species = df_clean['species']

print(species)

# Define custom colors: green for Gentoo, pink for Adelie, and blue for Chinstrap
species_colors = {'Gentoo': 'green', 'Adelie': 'pink', 'Chinstrap': 'blue'}
colors = species.map(species_colors)

# Create a 3D scatter plot of the first three principal components with species colored differently
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with custom species colors
sc = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=colors)

# Adding labels and titles
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Add a manual legend for species with custom colors
legend_labels = ['Gentoo', 'Adelie', 'Chinstrap']
legend_colors = ['green', 'pink', 'blue']
patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
ax.legend(patches, legend_labels, title="Species")

plt.show()