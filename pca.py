import numpy as np
import pandas as pd
from summary_statistics import df_clean, num_df
import matplotlib.pyplot as plt

# Get the attribute names and put them in a Numpy array
attributeNames = np.asarray(df_clean.columns)

# Step 1: Handle the non-numeric columns

# Convert the column containing species to Numpy array
species_arr = np.asarray(df_clean['species'])
# Extract each species
unique_species = np.unique(species_arr)
# Create a dictionary for mapping the species to integers
species_to_int = {species: idx for idx, species in enumerate(unique_species)}
# Encode the original array into integers using the dictionary
encoded_species_array = np.array([species_to_int[species] for species in species_arr])

# Let's do this for all non-numeric columns

island_arr = np.asarray(df_clean['island'])
unique_island = np.unique(island_arr)
island_to_int = {island: idx for idx, island in enumerate(unique_island)}
encoded_island_array = np.array([island_to_int[island] for island in island_arr])

sex_arr = np.asarray(df_clean['sex'])
unique_sex = np.unique(sex_arr)
sex_to_int = {sex: idx for idx, sex in enumerate(unique_sex)}
encoded_sex_array = np.array([sex_to_int[sex] for sex in sex_arr])

# Step 2: Let's standardize the numeric data so the mean is 0 
# and let's divide each value by the corresponding standard deviation so the variance in every dimension is 1

mean = num_df.mean()
std_dev = num_df.std()
standardized_num_data = (num_df - mean) / std_dev

# Step 3: Let's put all the data back together in a pandas dataframe
standardized_df = standardized_num_data
standardized_df['species'] = encoded_species_array
standardized_df['island'] = encoded_island_array
standardized_df['sex'] = encoded_sex_array

print(standardized_df)

# Step 4: Compute the covariance matrix using NumPy
cov_matrix = np.cov(standardized_df.T)  # Transpose because np.cov expects variables in rows, not columns

# Compute eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort the eigenvalues and corresponding eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Calculate the explained variance (eigenvalues represent variance)
explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
cumulative_variance_explained = np.cumsum(explained_variance_ratio)

# Plot the cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_explained, marker='o', linestyle='--', color='b')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Print the explained variance ratio for each component
print("Explained variance ratio by components:", explained_variance_ratio)
print("Cumulative explained variance:", cumulative_variance_explained)
"""
i = 1  # Index for the first attribute
j = 5  # Index for the second attribute

# Make a simple plot of the i'th attribute against the j'th attribute
#plt.plot(standardized_df.iloc[:, i], standardized_df.iloc[:, j], "o")  # Use iloc to index DataFrame columns

# Create another more fancy plot that includes legend, class labels,
# attribute names, and a title.
plt.figure()  # Initialize a new figure
plt.title("Penguin data")  # Add a title to the plot

C = len(unique_species)
y = encoded_species_array
for c in range(C):  # Assuming `C` is the number of unique classes
    # select indices belonging to class c:
    class_mask = (y == c)  # Create a boolean mask for class `c`
    
    # Plot the data points for class `c`
    plt.plot(standardized_df.iloc[class_mask, i], standardized_df.iloc[class_mask, j], "o", alpha=0.3)

# Add legend, labels, and axis names
plt.legend(unique_species)  # Assuming `classNames` is a list of class labels
plt.xlabel(attributeNames[i])  # Assuming `attributeNames` is a list of attribute names
plt.ylabel(attributeNames[j])

# Output result to screen
plt.show()
"""