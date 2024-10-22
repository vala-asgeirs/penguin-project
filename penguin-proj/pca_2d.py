import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from numpy import linalg as LA

# Open the file containing our data
df = pd.read_csv(Path("data/penguins_size.csv"))

# Replace . that are in our data with NaN
df.replace('.', np.nan, inplace=True)

# Drop all rows that contain NaN as a value, since we don't want to use those in our calculations
df = df.dropna()

# The numerical attributes we want to use are in columns 3, 4, 5, and 6
cols = range(2, 6)

# Get numerical data
num_df = df.iloc[:, cols]

# Extract species column separately for coloring later
species = df['species'].to_numpy()

# Convert numerical data to matrix format
matrix = num_df.to_numpy()

# Standardize the matrix: (value - mean) / std
norm_data = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)

# Compute X^T * X to get the covariance matrix
xt_x = np.dot(norm_data.T, norm_data)

# Perform Singular Value Decomposition (SVD)
U, S, Vh = np.linalg.svd(xt_x)

# Display the principal directions
print(f'U:\n{U}')
print(f'S:\n{S}')
print(f'Vh:\n{Vh}')
print(f'Principal direction 1: \n{Vh[:, 0]}')
print(f'Principal direction 2: \n{Vh[:, 1]}')

# Function to compute variance explained
def var_explained_n(n_components):
    return S[:n_components].sum() / S.sum()

# Variance explained by the first 2 components
print(f"Variance explained by first 2 components: {var_explained_n(2)}")

# Compute cumulative variance explained for each component
var_explained_each = np.array([var_explained_n(i) for i in range(1, 5)])

# Plot variance explained as a function of PCA components
fig, ax = plt.subplots(1, 1)
x_vals = range(1, 5)
ax.plot(x_vals, var_explained_each, '*-')
ax.set_title('Variance explained as a function of PCA components included')
ax.set_xlabel('N components')
ax.set_ylabel('Variance explained')
ax.set_xticks(np.arange(min(x_vals), max(x_vals) + 1, 1.0))
ax.set_ylim(0, 1)
plt.show()

# Project the standardized data onto the first two principal components
b1 = np.dot(norm_data, Vh.T[0])
b2 = np.dot(norm_data, Vh.T[1])

print(f'Shape of first principal component: {b1.shape}')

# Create an array to store the principal components
principal_components = np.array((b1, b2))

# Define colors for each species
species_colors = {'Gentoo': 'green', 'Adelie': 'pink', 'Chinstrap': 'blue'}
species_order = ['Gentoo', 'Adelie', 'Chinstrap']
colors = [species_colors[sp] for sp in species]

# Scatter plot of the first two principal components, colored by species
plt.figure(figsize=(10, 7))
for sp in species_order:
    plt.scatter(b1[species == sp], b2[species == sp], label=sp, color=species_colors[sp], s=50, alpha=0.7)

# Title and labels
#plt.title('The data projected onto the first 2 principal components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Species')
plt.show()
