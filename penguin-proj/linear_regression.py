import numpy as np
import pandas as pd
from standardize import standardized_df
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Define the predictor and target variables
X = standardized_df[['culmen_length_mm', 'culmen_depth_mm', 'body_mass_g']]
y = standardized_df['flipper_length_mm']

# Range of lambda (regularization parameter) values
lambda_values = np.logspace(-4, 4, 10)  # From 0.0001 to 10000 on a log scale

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Dictionary to store generalization errors for each lambda
gen_errors = {}

# Perform cross-validation for each lambda
for lmbda in lambda_values:
    # Store test errors for each fold
    test_errors = []
    
    # Iterate through each split
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train Ridge model with current lambda
        model = Ridge(alpha=lmbda)
        model.fit(X_train, y_train)
        
        # Predict and calculate test error
        y_pred = model.predict(X_test)
        test_error = mean_squared_error(y_test, y_pred)
        test_errors.append(test_error)
    
    # Average test error across folds for this lambda
    gen_errors[lmbda] = np.mean(test_errors)

# Find the optimal lambda with the lowest generalization error
optimal_lambda = min(gen_errors, key=gen_errors.get)
optimal_error = gen_errors[optimal_lambda]

# Train the final model on the entire dataset with the optimal lambda
best_model = Ridge(alpha=optimal_lambda)
best_model.fit(X, y)

# Print the optimal lambda and the corresponding model coefficients and intercept
print("Optimal Lambda:", optimal_lambda)
print("Lowest Generalization Error (MSE):", optimal_error)
print("Model Coefficients:", best_model.coef_)
print("Model Intercept:", best_model.intercept_)

import matplotlib.pyplot as plt

# Plotting the generalization error as a function of lambda
plt.figure(figsize=(8, 6))
plt.plot(lambda_values, list(gen_errors.values()), marker='o', color='#FF69B4')
plt.xscale('log')
plt.xlabel("Lambda (Regularization Parameter)")
plt.ylabel("Estimated Generalization Error (MSE)")
plt.grid(True)
plt.show()
