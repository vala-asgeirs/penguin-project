from summary_statistics import num_df, numAttributeNames

outliers_dict = dict()

for attr in numAttributeNames:
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = num_df[attr].quantile(0.25)
    Q3 = num_df[attr].quantile(0.75)

    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Calculate lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    outliers = num_df[(num_df[attr] < lower_bound) | (num_df[attr] > upper_bound)]

    if outliers.empty:
        print("There are no outliers for "+str(attr))
    else:
        outliers_dict[attr] = outliers
        print(str(attr)+" has outliers: "+str(outliers))

