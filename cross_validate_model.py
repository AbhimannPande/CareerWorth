<<<<<<< HEAD
from catboost import CatBoostRegressor, Pool, cv
import pandas as pd
import joblib

# Load data
X = joblib.load("X_processed.joblib")
y = joblib.load("y.joblib")

# Identify categorical feature indices
cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == "object"]

# Create Pool with categorical info
data_pool = Pool(data=X, label=y, cat_features=cat_features)

# Set model parameters
params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'RMSE',
    'verbose': 0
}

# Perform cross-validation
cv_results = cv(
    params=params,
    pool=data_pool,
    fold_count=5,
    shuffle=True,
    partition_random_seed=42,
    verbose=False,
    early_stopping_rounds=20
)

# Print results
print("✅ Cross-validation done")
print(f"📉 Best RMSE: {cv_results['test-RMSE-mean'].min():.4f}")
print(f"📈 Best R² Estimate: ~{1 - (cv_results['test-RMSE-mean'].min() / y.std())**2:.4f}")

=======
from catboost import CatBoostRegressor, Pool, cv
import pandas as pd
import joblib

# Load data
X = joblib.load("X_processed.joblib")
y = joblib.load("y.joblib")

# Identify categorical feature indices
cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == "object"]

# Create Pool with categorical info
data_pool = Pool(data=X, label=y, cat_features=cat_features)

# Set model parameters
params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'RMSE',
    'verbose': 0
}

# Perform cross-validation
cv_results = cv(
    params=params,
    pool=data_pool,
    fold_count=5,
    shuffle=True,
    partition_random_seed=42,
    verbose=False,
    early_stopping_rounds=20
)

# Print results
print("✅ Cross-validation done")
print(f"📉 Best RMSE: {cv_results['test-RMSE-mean'].min():.4f}")
print(f"📈 Best R² Estimate: ~{1 - (cv_results['test-RMSE-mean'].min() / y.std())**2:.4f}")

>>>>>>> d3f8614 (Initial commit with CareerWorth app)
