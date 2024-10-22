import numpy as np
from standardize import standardized_num_data
import matplotlib.pyplot as plt

cov_matrix = np.cov(standardized_num_data.T)  
# Transpose because np.cov expects variables in rows, not columns

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

plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')

plt.ylim(0,1)
plt.axhline(0.9, color='r')
plt.grid(True)
plt.show()

# Print the explained variance ratio for each component
print("Explained variance ratio by components:", explained_variance_ratio)
print("Cumulative explained variance:", cumulative_variance_explained)
