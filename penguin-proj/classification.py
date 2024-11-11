from standardize import standardized_df, species_to_int
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

# Define features and target
X = standardized_df.drop(columns=['species'])
y = standardized_df['species']

# Outer cross-validation setup (10-fold)
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Results storage
results = []

# Perform two-level cross-validation
for i, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
    y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
    
    # Baseline model (most frequent class)
    baseline_model = DummyClassifier(strategy='most_frequent')
    baseline_model.fit(X_train_outer, y_train_outer)
    baseline_error = 1 - accuracy_score(y_test_outer, baseline_model.predict(X_test_outer))
    
    # Inner cross-validation for hyperparameter tuning
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Multinomial Regression with hyperparameter tuning
    param_grid_multinomial = {'C': [0.01, 0.1, 1, 10, 100]}
    multinomial_search = GridSearchCV(LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42),
                                      param_grid_multinomial, cv=inner_cv, scoring='accuracy')
    multinomial_search.fit(X_train_outer, y_train_outer)
    multinomial_best_C = multinomial_search.best_params_['C']
    multinomial_test_error = 1 - accuracy_score(y_test_outer, multinomial_search.predict(X_test_outer))
    
    # KNN with hyperparameter tuning
    param_grid_knn = {'n_neighbors': [1, 3, 5, 7, 9]}
    knn_search = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=inner_cv, scoring='accuracy')
    knn_search.fit(X_train_outer, y_train_outer)
    knn_best_k = knn_search.best_params_['n_neighbors']
    knn_test_error = 1 - accuracy_score(y_test_outer, knn_search.predict(X_test_outer))
    
    # Store results for this outer fold
    results.append({
        "Outer Fold": i,
        "Multinomial Regression Best λ": 1 / multinomial_best_C,  # λ is the inverse of C
        "Multinomial Regression Test Error": multinomial_test_error,
        "KNN Best k": knn_best_k,
        "KNN Test Error": knn_test_error,
        "Baseline Test Error": baseline_error
    })

# Convert results to DataFrame and display the table
results_df = pd.DataFrame(results)
print(results_df)

