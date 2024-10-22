import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Open the file containing our data
df = pd.read_csv(Path("data/penguins_size.csv"))

# Replace . that are in our data with NaN
df.replace('.', np.nan, inplace=True)

# Drop all rows that contain NaN as a value, since we don't want to use those in our calculations
df = df.dropna()

# The numerical attributes we want to use are in columns 3, 4, 5 and 6
cols = range(2, 6)

# Get the names of the attributes we want to use
numAttributeNames = np.asarray(df.columns[cols])

# Create dictionary separated by species (species key gives data values)
species_to_df = {species: df[df["species"]==species] for species in df["species"].unique()}

# Create dictionary separated by species (species key gives numerical data values)
species_to_numd = {species: values.iloc[:, cols] for species,values in species_to_df.items()}

# Compile all numerical data for female adelie
num_df = species_to_df['Adelie'][species_to_df["Adelie"]['sex']=='FEMALE'].iloc[:, cols]

# Create a dictionary where each attribute name will be a key
num_attribute_dict = dict()

# Create another dictionary where each attribute name will be a key
perc_attribute_dict = dict()

print(f'Number of datapoints: {len(num_df)}')

covarience = num_df.cov()
print('\nCovariance Matrix:')
print(covarience)

correlation = num_df.corr()
print('\nCorrelation Matrix:')
print(correlation)

print('\nSummary Statistics:')
for attr in numAttributeNames:
    # Create a dictionary where the name of each statistic will be a key
    statistics_dict = dict()
    statistics_dict["Mean"] = num_df[attr].mean()
    statistics_dict["Varience"] = num_df[attr].var()
    statistics_dict["Standard Deviation"] = num_df[attr].std()

    # Create another dicitonary for the percintiles, where each statistic will be a key
    percintile_dict = dict()
    percintile_dict["Min"] = num_df[attr].min()
    percintile_dict["25% Quantile"] = num_df[attr].quantile(q=0.25)
    percintile_dict["Median"] = num_df[attr].median()
    percintile_dict["75% Quantile"] = num_df[attr].quantile(q=0.75)
    percintile_dict["Max"] = num_df[attr].max()
    percintile_dict["Range"] = num_df[attr].max() - num_df[attr].min()
    
    num_attribute_dict[attr] = statistics_dict
    perc_attribute_dict[attr] = percintile_dict

# Convert our dictionary of dictionaries to a Pandas dataframe that contains our statistics summary
statistics_df = pd.DataFrame(data=num_attribute_dict)

print(statistics_df)
print()

percentile_df = pd.DataFrame(data=perc_attribute_dict)

print(percentile_df)
print()

# Get the column that contains species in our data
species = df["species"]

# Count how many times each species appears in the data
speciesCount = species.value_counts()

print(speciesCount)
print()

# Get the column that contains islands in our data
islands = df["island"]

# Count how many times each island appears in the data
islandCount = islands.value_counts()

print(islandCount)
print()

# Get the column that contains sexes in our data
sex = df["sex"]

# Count how many times each value of sex appears
sexCount = sex.value_counts()

print(sexCount)
print()

# Plotting the histogram
plt.hist(df['body_mass_g'], bins=30, color='skyblue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Body Mass (g)')
plt.ylabel('Count')
plt.title('Distribution of Body Mass (g) over the whole dataset')
 
# Display the plot
plt.show()

# Create a dictionary to store the outlier indices for each attribute
outliers_dict = {}

# Loop over each numerical attribute to find outliers
for attr in numAttributeNames:
    # Calculate the first and third quartiles (Q1 and Q3)
    Q1 = num_df[attr].quantile(0.25)
    Q3 = num_df[attr].quantile(0.75)
    
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    
    # Define the outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find the outlier points
    outliers = num_df[(num_df[attr] < lower_bound) | (num_df[attr] > upper_bound)]
    
    # Store the outlier points in the dictionary
    outliers_dict[attr] = outliers

    # Print out the outliers for the current attribute
    print(f"Outliers for {attr}:")
    print(outliers)
    print()


