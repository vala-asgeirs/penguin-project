
import numpy as np
import pandas as pd
from explained_varience import eigenvectors
from summary_statistics import numAttributeNames
import matplotlib.pyplot as plt

# Create a DataFrame for the eigenvectors
eigenvectors_df = pd.DataFrame(eigenvectors, columns=[f'PC{i+1}' for i in range(eigenvectors.shape[1])], index=numAttributeNames)

# Plot the eigenvectors (loadings) for the first 3 principal components
eigenvectors_df[['PC1', 'PC2', 'PC3']].plot(kind='bar', figsize=(12, 6))
plt.xlabel('Features')
plt.ylabel('Contribution to the Principal Direction')
plt.xticks(rotation=0)
plt.legend(title='Principal Components')
plt.show()