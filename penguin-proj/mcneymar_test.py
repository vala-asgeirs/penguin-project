from standardize import standardized_df, species_to_int
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd
import numpy as np

# Define features and target
X = standardized_df.drop(columns=['species'])
y = standardized_df['species']

# Set up models with multinomial regression and KNN
multinomial_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=3)
baseline_model = DummyClassifier(strategy='most_frequent')

# Set up 10-fold cross-validation
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Generate out-of-fold predictions using cross_val_predict for each model
multinomial_preds = cross_val_predict(multinomial_model, X, y, cv=outer_cv)
knn_preds = cross_val_predict(knn_model, X, y, cv=outer_cv)
baseline_preds = cross_val_predict(baseline_model, X, y, cv=outer_cv)

# Function to perform McNemar's test and construct the correct contingency table
def perform_mcnemar_test(model_a_preds, model_b_preds, true_labels):
    # Determine correctness (True for correct, False for incorrect) for each model's predictions
    model_a_correct = model_a_preds == true_labels
    model_b_correct = model_b_preds == true_labels
    
    # Construct the contingency table based on correctness comparison
    contingency_table = pd.crosstab(model_a_correct, model_b_correct)
    print("Contingency Table:\n", contingency_table)
    
    # Ensure the contingency table is 2x2 for McNemar's test (fill missing values with 0)
    contingency_table = contingency_table.reindex(index=[False, True], columns=[False, True], fill_value=0)
    
    # Perform McNemar's test
    result = mcnemar(contingency_table, exact=False, correction=True)
    p_value = result.pvalue
    statistic = result.statistic
    
    # Extract values for n12 and n21
    n12 = contingency_table.loc[True, False]
    n21 = contingency_table.loc[False, True]
    
    # Compute a 95% confidence interval (approximate) for the difference
    diff = abs(n12 - n21)
    conf_interval = diff + 1.96 * np.sqrt(n12 + n21)  # 95% CI approximation
    
    return p_value, statistic, conf_interval

# Perform the McNemar's test for each pair of models

# 1. Multinomial Regression vs KNN
print("Multinomial Regression vs KNN")
p_value_multinomial_knn, statistic_multinomial_knn, ci_multinomial_knn = perform_mcnemar_test(multinomial_preds, knn_preds, y)
print("p-value:", p_value_multinomial_knn, "Statistic:", statistic_multinomial_knn, "Confidence Interval:", ci_multinomial_knn)

# 2. Multinomial Regression vs Baseline
print("\nMultinomial Regression vs Baseline")
p_value_multinomial_baseline, statistic_multinomial_baseline, ci_multinomial_baseline = perform_mcnemar_test(multinomial_preds, baseline_preds, y)
print("p-value:", p_value_multinomial_baseline, "Statistic:", statistic_multinomial_baseline, "Confidence Interval:", ci_multinomial_baseline)

# 3. KNN vs Baseline
print("\nKNN vs Baseline")
p_value_knn_baseline, statistic_knn_baseline, ci_knn_baseline = perform_mcnemar_test(knn_preds, baseline_preds, y)
print("p-value:", p_value_knn_baseline, "Statistic:", statistic_knn_baseline, "Confidence Interval:", ci_knn_baseline)
