import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from standardize import standardized_df
from sklearn.linear_model import Ridge

# Define the features and target
X = standardized_df[['culmen_length_mm', 'culmen_depth_mm', 'body_mass_g']]
y_true = standardized_df['flipper_length_mm']

# Use the model with the optimal lambda value
optimal_lambda = 2.78
model = Ridge(alpha=optimal_lambda)
model.fit(X, y_true)
y_pred = model.predict(X)

# Extract the learned coefficients
w1, w2, w3 = model.coef_
b = model.intercept_

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of true values with color representing culmen_depth_mm
sc = ax.scatter(X['culmen_length_mm'], X['body_mass_g'], y_true, 
                c=X['culmen_depth_mm'], cmap='viridis', label='True Flipper Length')
plt.colorbar(sc, label='Culmen Depth (standardized)')

# Generate predictions on a grid for the plot
culmen_length_range = np.linspace(X['culmen_length_mm'].min(), X['culmen_length_mm'].max(), 20)
body_mass_range = np.linspace(X['body_mass_g'].min(), X['body_mass_g'].max(), 20)
culmen_length_grid, body_mass_grid = np.meshgrid(culmen_length_range, body_mass_range)
culmen_depth_mean = X['culmen_depth_mm'].mean()  # Use the mean for a fixed depth
y_pred_grid = (w1 * culmen_length_grid + w2 * culmen_depth_mean + w3 * body_mass_grid + b)

# Surface plot for the model's predictions
ax.plot_surface(culmen_length_grid, body_mass_grid, y_pred_grid, color='orange', alpha=0.5, rstride=100, cstride=100)

# Labeling the axes
ax.set_xlabel('Culmen Length (standardized)')
ax.set_ylabel('Body Mass (standardized)')
ax.set_zlabel('Flipper Length (standardized)')
ax.set_title('3D Plot of Flipper Length Prediction with Optimal Ridge Model')

plt.show()
"""
import matplotlib.pyplot as plt
import numpy as np
from standardize import standardized_df
from sklearn.linear_model import Ridge

# Define the features and target
X = standardized_df[['culmen_length_mm', 'culmen_depth_mm', 'body_mass_g']]
y_true = standardized_df['flipper_length_mm']

# Use the model with the optimal lambda value
optimal_lambda = 2.78
model = Ridge(alpha=optimal_lambda)
model.fit(X, y_true)

# Mean values for the features (to hold constant)
culmen_length_mean = X['culmen_length_mm'].mean()
culmen_depth_mean = X['culmen_depth_mm'].mean()
body_mass_mean = X['body_mass_g'].mean()

# Generate a range of values for each feature
culmen_length_range = np.linspace(X['culmen_length_mm'].min(), X['culmen_length_mm'].max(), 100)
culmen_depth_range = np.linspace(X['culmen_depth_mm'].min(), X['culmen_depth_mm'].max(), 100)
body_mass_range = np.linspace(X['body_mass_g'].min(), X['body_mass_g'].max(), 100)

# Predict flipper length while varying each feature separately
# 1. Effect of Culmen Length
y_pred_culmen_length = model.predict(
    np.column_stack([culmen_length_range, np.full(100, culmen_depth_mean), np.full(100, body_mass_mean)])
)

# 2. Effect of Culmen Depth
y_pred_culmen_depth = model.predict(
    np.column_stack([np.full(100, culmen_length_mean), culmen_depth_range, np.full(100, body_mass_mean)])
)

# 3. Effect of Body Mass
y_pred_body_mass = model.predict(
    np.column_stack([np.full(100, culmen_length_mean), np.full(100, culmen_depth_mean), body_mass_range])
)

# Plotting the effects
plt.figure(figsize=(15, 5))

# Plot for Culmen Length
plt.subplot(1, 3, 1)
plt.plot(culmen_length_range, y_pred_culmen_length, color='blue')
plt.xlabel('Culmen Length (standardized)')
plt.ylabel('Predicted Flipper Length (standardized)')
plt.title('Effect of Culmen Length on Flipper Length')

# Plot for Culmen Depth
plt.subplot(1, 3, 2)
plt.plot(culmen_depth_range, y_pred_culmen_depth, color='green')
plt.xlabel('Culmen Depth (standardized)')
plt.ylabel('Predicted Flipper Length (standardized)')
plt.title('Effect of Culmen Depth on Flipper Length')

# Plot for Body Mass
plt.subplot(1, 3, 3)
plt.plot(body_mass_range, y_pred_body_mass, color='red')
plt.xlabel('Body Mass (standardized)')
plt.ylabel('Predicted Flipper Length (standardized)')
plt.title('Effect of Body Mass on Flipper Length')

plt.tight_layout()"""

