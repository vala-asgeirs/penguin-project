import numpy as np
import pandas as pd

# Open the file containing our data
df = pd.read_csv("../data/penguins_size.csv")

# Replace . that are in our data with NaN
df.replace('.', np.nan, inplace=True)

# Drop all rows that contain NaN as a value, since we don't want to use those in our calculations
df_clean = df.dropna()

# The numerical attributes we want to use are in columns 3, 4, 5 and 6
cols = range(2, 6)

# Get the names of the attributes we want to use
numAttributeNames = np.asarray(df_clean.columns[cols])

# Drop all columns exept the one we want to use from our dataframe
num_df = df_clean.iloc[:, cols]

# Create a dictionary where each attribute name will be a key
num_attribute_dict = dict()

# Create another dictionary where each attribute name will be a key
perc_attribute_dict = dict()

covarience = num_df.cov()
print(covarience)

correlation = num_df.corr()
print(correlation)

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
species = df_clean["species"]

# Count how many times each species appears in the data
speciesCount = species.value_counts()

print(speciesCount)
print()

# Get the column that contains islands in our data
islands = df_clean["island"]

# Count how many times each island appears in the data
islandCount = islands.value_counts()

print(islandCount)
print()

# Get the column that contains sexes in our data
sex = df_clean["sex"]

# Count how many times each value of sex appears
sexCount = sex.value_counts()

print(sexCount)
print()

