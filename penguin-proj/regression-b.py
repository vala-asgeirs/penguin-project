# ML project 2 - regression part b

import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
from scipy.stats import t

from dtuimldmtools import (
	draw_neural_net,
	train_neural_net,
	visualize_decision_boundary,
)

# Open the file containing our data
df = pd.read_csv("penguins_size.csv")

# Replace . that are in our data with NaN
df.replace('.', np.nan, inplace=True)

# Drop all rows that contain NaN as a value, since we don't want to use those in our calculations
df_clean = df.dropna()

# Choose columns of interest
culmen_length = df_clean.iloc[:,2]
culmen_depth = df_clean.iloc[:,3]
flipper_length = df_clean.iloc[:,4]
body_mass = df_clean.iloc[:,5]


# Standardize data and round to 3 decimals
culmen_length_standardized = (culmen_length - culmen_length.mean()) / culmen_length.std()
culmen_depth_standardized = (culmen_depth - culmen_depth.mean()) / culmen_depth.std()
flipper_length_standardized = (flipper_length - flipper_length.mean()) / flipper_length.std()
body_mass_standardized = (body_mass - body_mass.mean()) / body_mass.std()


# Turn them into numpy objects
culmen_depth_np = pd.Series.to_numpy(culmen_depth_standardized)
culmen_length_np = pd.Series.to_numpy(culmen_length_standardized)
flipper_length_np = pd.Series.to_numpy(flipper_length_standardized)
body_mass_np = pd.Series.to_numpy(body_mass_standardized)

# Stack into a matrix. Define X and y for regression
X = np.column_stack((culmen_length_np, culmen_depth_np, body_mass_np))
y = flipper_length_np
M, N = np.shape(X)

# Ordinary least squares linear regression
model = lm.LinearRegression()

# Fit the model to this data. So in this one, flipper length is predicted based on culmen length & depth
# Baseline model without regularization
model.fit(X, flipper_length_np)

# Lambdas (generalization parameters)
#lambdas = 10 ** np.linspace(-4,1,10)
#lambdas = 10 ** np.linspace(1,3,15)
lambdas = np.linspace(0,6,10)

# Splitting the data into test and train data, using 10-fold split
K1 = 10
CV_inner = KFold(n_splits = K1)

# For outer loop
K2 = 10
CV_outer = KFold(n_splits = K2)

# Train model
def train_model(data, target_data, train_index, lambdas):
	model = lm.Ridge(alpha=lambdas)
	model.fit(data[train_index], target_data[train_index])		# Check index
	return model

# Test model: find the test error based on the trained model. Test error = how far the predicted values are from the actual values
def test_model(data, target_data, test_index, trained_model):
	predicted_values = trained_model.predict(data[test_index]) 	## Check index
	test_error = mean_squared_error(target_data[test_index], predicted_values)
	return test_error

models = lambdas
test_errors = []
test_data_sizes = []
best_model_lambdas = []		# Lamba values corresponding the best models

for par_index, test_index in CV_outer.split(X):			# Outer fold
	inner_loop_counter = 0
	validation_errors = np.zeros([len(lambdas), K2])		# number of models, number of inner loops
	val_sizes = np.zeros([K2])
	par_size = len(par_index)								# Save |Dpar|

	for train_index, val_index in CV_inner.split(X):		# Inner fold
		for model_index, model in enumerate(models):		# enumerate gives both the index and the value
			trained_model = train_model(X, y, train_index, model)
			validation_error = test_model(X, y, val_index, trained_model)		# Now we calculate validation error instead of test error
			validation_errors[model_index, inner_loop_counter] = validation_error		# Save E_val_Ms,j

		val_sizes[inner_loop_counter] = (len(val_index))					# Save |Dval|
		inner_loop_counter += 1

	generalization_errors = (np.sum(validation_errors*val_sizes, axis=1)) / par_size		# Vector with generalization errors for each model

	best_model_index = np.argmin(generalization_errors)							# Find the best model (lowest gen error)
	best_model_trained = train_model(X, y, par_index, models[best_model_index])	# Train this model with all par data
	test_error = test_model(X, y, train_index, best_model_trained)				# Compute test error on test data
	test_errors.append(float(test_error))
	test_data_sizes.append(len(test_index))										# Save test data size
	best_model_lambdas.append(float(lambdas[best_model_index]))						# Lambda for the best model


final_generalization_error = np.sum(np.array(test_data_sizes) * np.array(test_errors)) / len(y)

print('Best lambdas', best_model_lambdas)
print('All test errors', test_errors)


# Implement baseline: mean of the train target data
def base_gen_error(data, target_data):      # data = X
	CV_base = KFold(n_splits = 10)
	test_errors = []
	for train_index, test_index in CV_base.split(data):
		train_mean = np.mean(target_data[train_index])
		test_errors.append(mean_squared_error(target_data[test_index], np.full((len(test_index)), train_mean)))
	return test_errors

baseline_test_errors = base_gen_error(X, y)
print('Baseline test errors', baseline_test_errors)


# ANN

def ANN_train_model(data, target_data, train_index, h_units):
    n_hidden_units = h_units        # Number of hidden units
    n_replicates = 1                # Number of networks trained in each k-fold
    max_iter = 3000

    # Set input size (M) based on the data
    X_train = torch.Tensor(data[train_index, :])
    M = X_train.shape[1]  # Number of features
    
    # Define the neural network model
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to n hidden units
        torch.nn.Tanh(),                     # Activation function
        torch.nn.Linear(n_hidden_units, 1)   # n hidden units to 1 output neuron
    )

    # Define the loss function
    loss_fn = torch.nn.MSELoss()

    # Prepare the target tensor and ensure it's in the correct shape
    y_train = torch.Tensor(target_data[train_index]).unsqueeze(1)

    # Train the network using train_neural_net function
    net, final_loss, learning_curve = train_neural_net(
        model,
        loss_fn,
        X=X_train,
        y=y_train,
        n_replicates=n_replicates,
        max_iter=max_iter
    )

    return net

def ANN_test_model(data, target_data, test_index, net):
    # Prepare test data and ensure proper shape
    X_test = torch.Tensor(data[test_index, :])
    y_test = torch.Tensor(target_data[test_index]).unsqueeze(1)  # Shape as (n_samples, 1)

    # Forward pass: get network's predictions for the test set
    y_test_est = net(X_test)

    # Calculate Mean Squared Error (MSE)
    se = (y_test_est - y_test) ** 2  # Squared errors
    mse = se.mean().item()           # Calculate MSE by averaging squared errors

    return mse


# 2-fold CV for ANN

# Splitting the data into test and train data, using 10-fold split
K1 = 10
CV_inner = KFold(n_splits = K1)

# For outer loop
K2 = 10
CV_outer = KFold(n_splits = K2)

lambdas = np.array([1,5, 10, 15, 20])

models = lambdas
test_errors = []
test_data_sizes = []
best_model_lambdas = []		# Lamba values corresponding the best models

for par_index, test_index in CV_outer.split(X):			# Outer fold
	inner_loop_counter = 0
	validation_errors = np.zeros([len(lambdas), K2])		# number of models, number of inner loops
	val_sizes = np.zeros([K2])
	par_size = len(par_index)								# Save |Dpar|

	for train_index, val_index in CV_inner.split(X):		# Inner fold
		for model_index, model in enumerate(models):		# enumerate gives both the index and the value
			trained_model = ANN_train_model(X, y, train_index, model)
			validation_error = ANN_test_model(X, y, val_index, trained_model)		# Now we calculate validation error instead of test error
			validation_errors[model_index, inner_loop_counter] = validation_error		# Save E_val_Ms,j

		val_sizes[inner_loop_counter] = (len(val_index))					# Save |Dval|
		inner_loop_counter += 1

	generalization_errors = (np.sum(validation_errors*val_sizes, axis=1)) / par_size		# Vector with generalization errors for each model

	best_model_index = np.argmin(generalization_errors)							# Find the best model (lowest gen error)
	best_model_trained = ANN_train_model(X, y, par_index, models[best_model_index])	# Train this model with all par data
	test_error = ANN_test_model(X, y, train_index, best_model_trained)				# Compute test error on test data
	test_errors.append(test_error)
	test_data_sizes.append(len(test_index))										# Save test data size
	best_model_lambdas.append(lambdas[best_model_index])						# Lambda for the best model


print('ANN: Best h', best_model_lambdas)
print('ANN: All test errors', test_errors)


# Setup II: correlated t-test for cross validation

# ANN vs linear model
nu = 10-1		# degree of freedom
r = np.empty((10))		# array to store differencen in performance for each fold
K = KFold(n_splits = 10, shuffle = True)
count = -1
best_lambda = 0
best_h = 5

for train_index, test_index in K.split(X):
	count += 1
	lin_model = train_model(X, y, train_index, best_lambda)
	net_model = ANN_train_model(X, y, train_index, best_h)
	r[count] = test_model(X, y, test_index, lin_model) - ANN_test_model(X, y, test_index, net_model)		# Difference between performance

r_hat = np.mean(r)		# Mean of performance differences across folds
s_hat = np.std(r)		# STD of performance differences across folds
s_tilde = np.sqrt((1/10 + 1/(10-1))) * s_hat		# STD of the mean difference

sig_level = 0.05
t_critical = t.ppf(1 - sig_level / 2, nu)  # critical t-value for 95% confidence interval
conf_int = [r_hat - t_critical * s_tilde, r_hat + t_critical * s_tilde]
t_hat = r_hat/s_tilde		# t-statistic

# p-value for the observed t-statistic
pval = t.cdf(t_hat, nu)
print('LM x ANN', 'CI:', conf_int, 'p-value:', pval)

#----------------------------------------------------------------------------------------------
# ANN vs baseline
nu = 10-1		# degree of freedom
r = np.empty((10))		# array to store differencen in performance for each fold
K = KFold(n_splits = 10, shuffle = True)
count = -1
best_lambda = 0
best_h = 5

for train_index, test_index in K.split(X):
	count += 1
	baseline_model = np.mean(y[train_index])
	net_model = ANN_train_model(X, y, train_index, best_h)
	r[count] = mean_squared_error(y[test_index], np.full((len(test_index)), baseline_model)) - ANN_test_model(X, y, test_index, net_model)		# Difference between performance

r_hat = np.mean(r)		# Mean of performance differences across folds
s_hat = np.std(r)		# STD of performance differences across folds
s_tilde = np.sqrt((1/10 + 1/(10-1))) * s_hat		# STD of the mean difference

sig_level = 0.05
t_critical = t.ppf(1 - sig_level / 2, nu)  # critical t-value for 95% confidence interval
conf_int = [r_hat - t_critical * s_tilde, r_hat + t_critical * s_tilde]
t_hat = r_hat/s_tilde		# t-statistic

# p-value for the observed t-statistic
pval = t.cdf(t_hat, nu)
print('ANN x baseline', 'CI:', conf_int, 'p-value:', pval)

#-------------------------------------------------------------------------------------------------
# LM x baseline
nu = 10-1		# degree of freedom
r = np.empty((10))		# array to store differencen in performance for each fold
K = KFold(n_splits = 10, shuffle = True)
count = -1
best_lambda = 0
best_h = 5

for train_index, test_index in K.split(X):
	count += 1
	lin_model = train_model(X, y, train_index, best_lambda)
	baseline_model = np.mean(y[train_index])
	r[count] = mean_squared_error(y[test_index], np.full((len(test_index)), baseline_model)) - test_model(X, y, test_index, lin_model)		# Difference between performance

r_hat = np.mean(r)		# Mean of performance differences across folds
s_hat = np.std(r)		# STD of performance differences across folds
s_tilde = np.sqrt((1/10 + 1/(10-1))) * s_hat		# STD of the mean difference

sig_level = 0.05
t_critical = t.ppf(1 - sig_level / 2, nu)  # critical t-value for 95% confidence interval
conf_int = [r_hat - t_critical * s_tilde, r_hat + t_critical * s_tilde]
t_hat = r_hat/s_tilde		# t-statistic

# p-value for the observed t-statistic
pval = t.cdf(t_hat, nu)
print('LM x baseline', 'CI:', conf_int, 'p-value:', pval)