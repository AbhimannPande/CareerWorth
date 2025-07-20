<<<<<<< HEAD
# preprocess.py
import pandas as pd
import numpy as np
from pathlib import Path
from config import *  # Uses the config we just created

def preprocess_data(df):
    """
    Applies all data preprocessing steps:
    1. Creates derived features
    2. Handles missing values
    3. Applies log transformations
    """
    # Create derived features
    for feature_name, func in DERIVED_FEATURES.items():
        df[feature_name] = func(df)
    
    # Log transform monetary features (skip if negative values exist)
    for feat in ['capital-gain', 'capital-loss', TARGET]:
        if feat in df.columns:
            if (df[feat] >= 0).all():  # Only log transform if all values are non-negative
                df[feat] = np.log1p(df[feat])
            else:
                print(f"Warning: {feat} contains negative values, skipping log transform")
    
    return df

def load_and_process():
    """Loads raw data and applies preprocessing"""
    # Verify raw data exists
    if not RAW_DATA.exists():
        available_files = [f.name for f in DATA_DIR.glob('*') if f.is_file()]
        raise FileNotFoundError(
            f"Raw data file not found at {RAW_DATA}\n"
            f"Available files in {DATA_DIR}: {available_files}"
        )
    
    # Load data with proper dtype specification
    dtype_spec = {
        'workclass': 'category',
        'education': 'category',
        'marital-status': 'category',
        'occupation': 'category',
        'relationship': 'category',
        'gender': 'category',
        'native-country': 'category'
    }
    
    try:
        df = pd.read_csv(
            RAW_DATA,
            dtype=dtype_spec,
            na_values=[' ?', '?', 'NA', 'N/A']
        )
    except Exception as e:
        raise ValueError(f"Error reading {RAW_DATA}: {str(e)}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("Warning: Missing values detected:")
        print(missing[missing > 0])
        # Fill missing categoricals with 'Unknown'
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].cat.add_categories(['Unknown']).fillna('Unknown')
    
    # Process the data
    processed_df = preprocess_data(df)
    
    # Ensure all expected columns exist
    for col in ALL_FEATURES + [TARGET]:
        if col not in processed_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Save processed data
    processed_df.to_csv(PROCESSED_DATA, index=False)
    return processed_df

if __name__ == "__main__":
    print(f"Starting preprocessing of {RAW_DATA.name}...")
    try:
        df = load_and_process()
        print(f"Successfully processed data. Shape: {df.shape}")
        print(f"Saved to: {PROCESSED_DATA}")
        print("\nSample of processed data:")
        print(df.head())
        
        # Print summary of transformed target
        if TARGET in df.columns:
            print("\nTarget variable summary:")
            print(f"Original range: {np.expm1(df[TARGET]).min():.2f} to {np.expm1(df[TARGET]).max():.2f}")
            print(f"Log-transformed range: {df[TARGET].min():.2f} to {df[TARGET].max():.2f}")
            
    except Exception as e:
=======
# preprocess.py
import pandas as pd
import numpy as np
from pathlib import Path
from config import *  # Uses the config we just created

def preprocess_data(df):
    """
    Applies all data preprocessing steps:
    1. Creates derived features
    2. Handles missing values
    3. Applies log transformations
    """
    # Create derived features
    for feature_name, func in DERIVED_FEATURES.items():
        df[feature_name] = func(df)
    
    # Log transform monetary features (skip if negative values exist)
    for feat in ['capital-gain', 'capital-loss', TARGET]:
        if feat in df.columns:
            if (df[feat] >= 0).all():  # Only log transform if all values are non-negative
                df[feat] = np.log1p(df[feat])
            else:
                print(f"Warning: {feat} contains negative values, skipping log transform")
    
    return df

def load_and_process():
    """Loads raw data and applies preprocessing"""
    # Verify raw data exists
    if not RAW_DATA.exists():
        available_files = [f.name for f in DATA_DIR.glob('*') if f.is_file()]
        raise FileNotFoundError(
            f"Raw data file not found at {RAW_DATA}\n"
            f"Available files in {DATA_DIR}: {available_files}"
        )
    
    # Load data with proper dtype specification
    dtype_spec = {
        'workclass': 'category',
        'education': 'category',
        'marital-status': 'category',
        'occupation': 'category',
        'relationship': 'category',
        'gender': 'category',
        'native-country': 'category'
    }
    
    try:
        df = pd.read_csv(
            RAW_DATA,
            dtype=dtype_spec,
            na_values=[' ?', '?', 'NA', 'N/A']
        )
    except Exception as e:
        raise ValueError(f"Error reading {RAW_DATA}: {str(e)}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("Warning: Missing values detected:")
        print(missing[missing > 0])
        # Fill missing categoricals with 'Unknown'
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].cat.add_categories(['Unknown']).fillna('Unknown')
    
    # Process the data
    processed_df = preprocess_data(df)
    
    # Ensure all expected columns exist
    for col in ALL_FEATURES + [TARGET]:
        if col not in processed_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Save processed data
    processed_df.to_csv(PROCESSED_DATA, index=False)
    return processed_df

if __name__ == "__main__":
    print(f"Starting preprocessing of {RAW_DATA.name}...")
    try:
        df = load_and_process()
        print(f"Successfully processed data. Shape: {df.shape}")
        print(f"Saved to: {PROCESSED_DATA}")
        print("\nSample of processed data:")
        print(df.head())
        
        # Print summary of transformed target
        if TARGET in df.columns:
            print("\nTarget variable summary:")
            print(f"Original range: {np.expm1(df[TARGET]).min():.2f} to {np.expm1(df[TARGET]).max():.2f}")
            print(f"Log-transformed range: {df[TARGET].min():.2f} to {df[TARGET].max():.2f}")
            
    except Exception as e:
>>>>>>> d3f8614 (Initial commit with CareerWorth app)
        print(f"Error during preprocessing: {str(e)}")