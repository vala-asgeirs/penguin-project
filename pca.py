import numpy as np
import pandas as pd
from summary_statistics import df_clean, num_df

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

# 

