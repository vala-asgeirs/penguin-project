import numpy as np
import pandas as pd
from summary_statistics import *
from pca import * 

outliers_dict = dict()

for attr in numAttributeNames:
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df_clean[attr].quantile(0.25)
    Q3 = df_clean[attr].quantile(0.75)

    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Calculate lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    outliers = df[(df[attr] < lower_bound) | (df[attr] > upper_bound)]
    print("outliers: "+str(outliers)+" for: "+str(attr))
    outliers_dict[attr] = outliers

