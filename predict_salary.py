<<<<<<< HEAD
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor, Pool

def get_model_features():
    """Extract feature requirements directly from the model with types"""
    model = CatBoostRegressor()
    model.load_model("catboost_salary_model.cbm")
    
    # Get categorical features indices from the model
    cat_indices = model.get_cat_feature_indices()
    feature_names = model.feature_names_
    
    # Categorize features
    categoricals = [feature_names[i] for i in cat_indices]
    numericals = [f for f in feature_names if f not in categoricals]
    
    return {
        'all_features': feature_names,
        'categorical': categoricals,
        'numerical': numericals
    }

def preprocess_input(df_raw, feature_spec):
    """Prepare input data matching exact model requirements"""
    # 1. Ensure all features exist
    for feature in feature_spec['all_features']:
        if feature not in df_raw.columns:
            if feature in feature_spec['numerical']:
                df_raw[feature] = 0  # Default for numerical
            else:
                df_raw[feature] = 'Unknown'  # Default for categorical
    
    # 2. Create derived features
    if 'total_capital' in feature_spec['all_features']:
        if all(col in df_raw.columns for col in ['capital-gain', 'capital-loss']):
            df_raw['total_capital'] = df_raw['capital-gain'] - df_raw['capital-loss']
        elif 'total_capital' not in df_raw.columns:
            df_raw['total_capital'] = 0
    
    # 3. Convert types
    for col in feature_spec['numerical']:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
    
    for col in feature_spec['categorical']:
        df_raw[col] = df_raw[col].astype(str).fillna('Unknown')
    
    return df_raw[feature_spec['all_features']]

def predict_salary(input_data):
    try:
        # Get model's exact feature requirements
        feature_spec = get_model_features()
        
        # Prepare input DataFrame
        if isinstance(input_data, dict):
            df_input = pd.DataFrame([input_data])
        else:
            df_input = input_data.copy()
        
        # Preprocess to match model requirements
        X_processed = preprocess_input(df_input, feature_spec)
        
        # Get categorical indices
        cat_indices = [
            i for i, col in enumerate(X_processed.columns) 
            if col in feature_spec['categorical']
        ]
        
        # Make prediction
        model = CatBoostRegressor()
        model.load_model("catboost_salary_model.cbm")
        
        pool = Pool(
            data=X_processed,
            cat_features=cat_indices,
            feature_names=list(X_processed.columns)
        )
        
        return round(model.predict(pool)[0], 2)
    
    except Exception as e:
        required = feature_spec['all_features']
        provided = list(input_data.keys()) if isinstance(input_data, dict) else list(input_data.columns)
        raise ValueError(
            f"Prediction failed: {str(e)}\n"
            f"Model requires: {required}\n"
            f"Provided features: {provided}\n"
            f"Missing features: {set(required) - set(provided)}"
        )

# Example usage with ALL required features
if __name__ == "__main__":
    # Complete input including the new 'currency' feature
    test_input = {
        'age': 32,
        'workclass': 'Private',
        'education': 'Bachelors',
        'marital-status': 'Single',
        'occupation': 'Engineer',
        'relationship': 'Not-in-family',
        'gender': 'Male',
        'native-country': 'United-States',
        'hours-per-week': 40,
        'capital-gain': 5000,
        'capital-loss': 1000,
        'currency': 'USD'  # New required categorical feature
    }
    
    try:
        salary = predict_salary(test_input)
        print(f"Predicted Salary: {salary:,.2f} {test_input['currency']}")
    except Exception as e:
=======
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor, Pool

def get_model_features():
    """Extract feature requirements directly from the model with types"""
    model = CatBoostRegressor()
    model.load_model("catboost_salary_model.cbm")
    
    # Get categorical features indices from the model
    cat_indices = model.get_cat_feature_indices()
    feature_names = model.feature_names_
    
    # Categorize features
    categoricals = [feature_names[i] for i in cat_indices]
    numericals = [f for f in feature_names if f not in categoricals]
    
    return {
        'all_features': feature_names,
        'categorical': categoricals,
        'numerical': numericals
    }

def preprocess_input(df_raw, feature_spec):
    """Prepare input data matching exact model requirements"""
    # 1. Ensure all features exist
    for feature in feature_spec['all_features']:
        if feature not in df_raw.columns:
            if feature in feature_spec['numerical']:
                df_raw[feature] = 0  # Default for numerical
            else:
                df_raw[feature] = 'Unknown'  # Default for categorical
    
    # 2. Create derived features
    if 'total_capital' in feature_spec['all_features']:
        if all(col in df_raw.columns for col in ['capital-gain', 'capital-loss']):
            df_raw['total_capital'] = df_raw['capital-gain'] - df_raw['capital-loss']
        elif 'total_capital' not in df_raw.columns:
            df_raw['total_capital'] = 0
    
    # 3. Convert types
    for col in feature_spec['numerical']:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
    
    for col in feature_spec['categorical']:
        df_raw[col] = df_raw[col].astype(str).fillna('Unknown')
    
    return df_raw[feature_spec['all_features']]

def predict_salary(input_data):
    try:
        # Get model's exact feature requirements
        feature_spec = get_model_features()
        
        # Prepare input DataFrame
        if isinstance(input_data, dict):
            df_input = pd.DataFrame([input_data])
        else:
            df_input = input_data.copy()
        
        # Preprocess to match model requirements
        X_processed = preprocess_input(df_input, feature_spec)
        
        # Get categorical indices
        cat_indices = [
            i for i, col in enumerate(X_processed.columns) 
            if col in feature_spec['categorical']
        ]
        
        # Make prediction
        model = CatBoostRegressor()
        model.load_model("catboost_salary_model.cbm")
        
        pool = Pool(
            data=X_processed,
            cat_features=cat_indices,
            feature_names=list(X_processed.columns)
        )
        
        return round(model.predict(pool)[0], 2)
    
    except Exception as e:
        required = feature_spec['all_features']
        provided = list(input_data.keys()) if isinstance(input_data, dict) else list(input_data.columns)
        raise ValueError(
            f"Prediction failed: {str(e)}\n"
            f"Model requires: {required}\n"
            f"Provided features: {provided}\n"
            f"Missing features: {set(required) - set(provided)}"
        )

# Example usage with ALL required features
if __name__ == "__main__":
    # Complete input including the new 'currency' feature
    test_input = {
        'age': 32,
        'workclass': 'Private',
        'education': 'Bachelors',
        'marital-status': 'Single',
        'occupation': 'Engineer',
        'relationship': 'Not-in-family',
        'gender': 'Male',
        'native-country': 'United-States',
        'hours-per-week': 40,
        'capital-gain': 5000,
        'capital-loss': 1000,
        'currency': 'USD'  # New required categorical feature
    }
    
    try:
        salary = predict_salary(test_input)
        print(f"Predicted Salary: {salary:,.2f} {test_input['currency']}")
    except Exception as e:
>>>>>>> d3f8614 (Initial commit with CareerWorth app)
        print(f"Error: {e}")