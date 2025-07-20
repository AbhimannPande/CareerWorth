<<<<<<< HEAD
import pandas as pd
import joblib
from catboost import CatBoostRegressor, Pool
from config import *
from pathlib import Path

def train_model():
    # Load processed data
    processed_df = pd.read_csv(PROCESSED_DATA)
    
    # Prepare features and target
    X = processed_df[ALL_FEATURES]
    y = processed_df[TARGET]
    
    # Create CatBoost Pool (optimized data structure)
    train_pool = Pool(
        data=X,
        label=y,
        cat_features=CATEGORICAL_FEATURES
    )
    
    # Initialize and train model
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train_pool)
    
    # Save model and feature info
    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(MODEL_DIR / "salary_model.cbm")
    
    feature_info = {
        'feature_order': model.feature_names_,
        'categorical_cols': CATEGORICAL_FEATURES,
        'derived_features': list(DERIVED_FEATURES.keys())
    }
    joblib.dump(feature_info, MODEL_DIR / "feature_info.joblib")
    
    return model

if __name__ == "__main__":
    print("🚀 Training model...")
    try:
        model = train_model()
        print(f"✅ Model successfully trained and saved to {MODEL_DIR}")
        print("📊 Feature importance:")
        print(model.get_feature_importance(prettified=True))
    except Exception as e:
=======
import pandas as pd
import joblib
from catboost import CatBoostRegressor, Pool
from config import *
from pathlib import Path

def train_model():
    # Load processed data
    processed_df = pd.read_csv(PROCESSED_DATA)
    
    # Prepare features and target
    X = processed_df[ALL_FEATURES]
    y = processed_df[TARGET]
    
    # Create CatBoost Pool (optimized data structure)
    train_pool = Pool(
        data=X,
        label=y,
        cat_features=CATEGORICAL_FEATURES
    )
    
    # Initialize and train model
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train_pool)
    
    # Save model and feature info
    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(MODEL_DIR / "salary_model.cbm")
    
    feature_info = {
        'feature_order': model.feature_names_,
        'categorical_cols': CATEGORICAL_FEATURES,
        'derived_features': list(DERIVED_FEATURES.keys())
    }
    joblib.dump(feature_info, MODEL_DIR / "feature_info.joblib")
    
    return model

if __name__ == "__main__":
    print("🚀 Training model...")
    try:
        model = train_model()
        print(f"✅ Model successfully trained and saved to {MODEL_DIR}")
        print("📊 Feature importance:")
        print(model.get_feature_importance(prettified=True))
    except Exception as e:
>>>>>>> d3f8614 (Initial commit with CareerWorth app)
        print(f"❌ Training failed: {str(e)}")