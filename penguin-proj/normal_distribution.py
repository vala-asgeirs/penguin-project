from standardize import standardized_num_data
import matplotlib.pyplot as plt

num_plot = 1
for feature in standardized_num_data.columns:
    plt.figure(figsize=(7, 5))
    
    # Create a histogram for each feature
    plt.hist(standardized_num_data[feature], bins=20, color='#FF69B4', edgecolor='black')
    
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    print("Plot "+str(num_plot)+" is of "+str(feature))
    num_plot+=1
 
    plt.grid(visible=False)  # Remove grid lines, Tufte's guidelines ;)
    plt.show()
