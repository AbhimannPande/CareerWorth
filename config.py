<<<<<<< HEAD
# config.py
from pathlib import Path

# =============================================
# Directory Configuration with Auto-Creation
# =============================================

# Create directories if they don't exist
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Creates both 'data' and parent directories if needed
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =============================================
# File Paths
# =============================================
RAW_DATA = DATA_DIR / "employee_salaries_custom.csv"        # Your input CSV file
PROCESSED_DATA = DATA_DIR / "processed_data.csv"

# =============================================
# Feature Configuration
# =============================================

# Target variable
TARGET = "salary"

# Categorical features (exactly as they appear in your CSV)
CATEGORICAL_FEATURES = [
    'workclass',
    'education',
    'marital-status',  # Note the hyphen
    'occupation',
    'relationship',
    'gender',
    'native-country'   # Note the hyphen
]

# Numerical features
NUMERICAL_FEATURES = [
    'age',
    'hours-per-week',
    'capital-gain',    # Will be log-transformed
    'capital-loss'     # Will be log-transformed
]

# Derived features (created during preprocessing)
DERIVED_FEATURES = {
    'total_capital': lambda df: df['capital-gain'] - df['capital-loss']
}

# =============================================
# Model Configuration
# =============================================
CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'loss_function': 'RMSE',
    'verbose': 100,
    'random_seed': 42,
    'cat_features': CATEGORICAL_FEATURES  # Explicitly specify categorical features
}

# =============================================
# Complete Feature List (Automatically generated)
# =============================================
ALL_FEATURES = NUMERICAL_FEATURES + list(DERIVED_FEATURES.keys()) + CATEGORICAL_FEATURES

# =============================================
# Validation
# =============================================
if __name__ == "__main__":
    print("Configuration Validation:")
    print(f"Data Directory: {DATA_DIR} (exists: {DATA_DIR.exists()})")
    print(f"Model Directory: {MODEL_DIR} (exists: {MODEL_DIR.exists()})")
    print("\nFeature Counts:")
    print(f"Categorical: {len(CATEGORICAL_FEATURES)}")
    print(f"Numerical: {len(NUMERICAL_FEATURES)}")
    print(f"Derived: {len(DERIVED_FEATURES)}")
=======
# config.py
from pathlib import Path

# =============================================
# Directory Configuration with Auto-Creation
# =============================================

# Create directories if they don't exist
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Creates both 'data' and parent directories if needed
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =============================================
# File Paths
# =============================================
RAW_DATA = DATA_DIR / "employee_salaries_custom.csv"        # Your input CSV file
PROCESSED_DATA = DATA_DIR / "processed_data.csv"

# =============================================
# Feature Configuration
# =============================================

# Target variable
TARGET = "salary"

# Categorical features (exactly as they appear in your CSV)
CATEGORICAL_FEATURES = [
    'workclass',
    'education',
    'marital-status',  # Note the hyphen
    'occupation',
    'relationship',
    'gender',
    'native-country'   # Note the hyphen
]

# Numerical features
NUMERICAL_FEATURES = [
    'age',
    'hours-per-week',
    'capital-gain',    # Will be log-transformed
    'capital-loss'     # Will be log-transformed
]

# Derived features (created during preprocessing)
DERIVED_FEATURES = {
    'total_capital': lambda df: df['capital-gain'] - df['capital-loss']
}

# =============================================
# Model Configuration
# =============================================
CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'loss_function': 'RMSE',
    'verbose': 100,
    'random_seed': 42,
    'cat_features': CATEGORICAL_FEATURES  # Explicitly specify categorical features
}

# =============================================
# Complete Feature List (Automatically generated)
# =============================================
ALL_FEATURES = NUMERICAL_FEATURES + list(DERIVED_FEATURES.keys()) + CATEGORICAL_FEATURES

# =============================================
# Validation
# =============================================
if __name__ == "__main__":
    print("Configuration Validation:")
    print(f"Data Directory: {DATA_DIR} (exists: {DATA_DIR.exists()})")
    print(f"Model Directory: {MODEL_DIR} (exists: {MODEL_DIR.exists()})")
    print("\nFeature Counts:")
    print(f"Categorical: {len(CATEGORICAL_FEATURES)}")
    print(f"Numerical: {len(NUMERICAL_FEATURES)}")
    print(f"Derived: {len(DERIVED_FEATURES)}")
>>>>>>> d3f8614 (Initial commit with CareerWorth app)
    print(f"Total: {len(ALL_FEATURES)} features + 1 target ({TARGET})")