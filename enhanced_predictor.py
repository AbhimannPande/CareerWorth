<<<<<<< HEAD
# enhanced_predictor.py (Fixed Version)
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import joblib
from pathlib import Path

class EnhancedSalaryPredictor:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_info = None
        self.load_artifacts()
        
        # Configuration
        self.country_floors = {
            'USA': {'default': 15000, 'Handlers-cleaners': 18000, 'Craft-repair': 25000},
            'India': {
                'default': 200000,
                'Adm-clerical': 180000,
                'Handlers-cleaners': 250000,
                'Craft-repair': 250000,
                'Tech-support': 300000
            },
            'Germany': {'default': 12000}
        }
        
        self.occupation_baselines = {
            'Exec-managerial': 1500000,
            'Tech-support': 600000,
            'Adm-clerical': 360000,
            'Handlers-cleaners': 300000,
            'Craft-repair': 450000
        }
        
        self.marital_premiums = {
            'Married-civ-spouse': 1.08,
            'Never-married': 0.98,
            'Divorced': 1.0
        }
        
        self.MAX_AGE = 100
        self.MAX_HOURS = {
            'default': 40,
            'senior': 30,
            'elderly': 20,
            'centenarian': 10
        }

        self.ELDERLY_MINIMUMS = {
            'India': {
                'Handlers-cleaners': 220000,
                'Craft-repair': 200000,
                'default': 180000
            }
        }
        
        self.MIN_SALARY_RATIOS = {
            'senior': 0.75,
            'elderly': 0.65,
            'centenarian': 0.55
        }
        
        self.DECAY_RATES = {
            'senior': 0.008,
            'elderly': 0.012,
            'centenarian': 0.016
        }

        self.OCCUPATION_FLOOR_MULTIPLIERS = {
            'Handlers-cleaners': 1.4,
            'Craft-repair': 1.6,
            'Adm-clerical': 1.2,
            'Tech-support': 1.3,
            'default': 1.1
        }

    def load_artifacts(self):
        self.model = CatBoostRegressor()
        self.model.load_model(self.model_dir / "salary_model.cbm")
        self.feature_info = joblib.load(self.model_dir / "feature_info.joblib")

    def preprocess_input(self, input_data):
        self._validate_age(input_data['age'])
        input_data['hours-per-week'] = self._adjust_hours_for_age(
            input_data['hours-per-week'], 
            input_data['age']
        )
        
        df = pd.DataFrame([input_data])
        
        for feat in self.feature_info.get('derived_features', []):
            if feat == 'total_capital':
                df[feat] = df['capital-gain'] - df['capital-loss']
            elif feat.startswith('log_'):
                src_col = feat[4:]
                df[src_col] = np.log1p(df[src_col])
        
        df['capital-boost'] = np.log1p(df['capital-gain']) * 0.6
        df['experience-bonus'] = np.sqrt(df['age'] - 18) * 1200
        
        for feature in self.feature_info['feature_order'] + ['capital-boost', 'experience-bonus']:
            if feature not in df.columns:
                df[feature] = 0
                
        return df[self.feature_info['feature_order'] + ['capital-boost', 'experience-bonus']]

    def _validate_age(self, age):
        if not (18 <= age <= self.MAX_AGE):
            raise ValueError(
                f"Age {age} invalid. Must be 18-{self.MAX_AGE}. "
                f"Use 'force=True' to override (not recommended)."
            )

    def _adjust_hours_for_age(self, hours, age):
        if age >= 90:
            return min(hours, self.MAX_HOURS['centenarian'])
        elif age >= 80:
            return min(hours, self.MAX_HOURS['elderly'])
        elif age >= 65:
            return min(hours, self.MAX_HOURS['senior'])
        return min(hours, self.MAX_HOURS['default'])

    def predict(self, input_data, force=False):
        try:
            if not force:
                self._validate_age(input_data['age'])
            
            input_df = self.preprocess_input(input_data)
            pool = Pool(input_df, cat_features=self.feature_info['categorical_cols'])
            raw_pred = np.expm1(self.model.predict(pool)[0])
            
            adjusted = self._apply_country_floor(raw_pred, input_data)
            adjusted = self._apply_occupation_baseline(adjusted, input_data)
            adjusted = self._apply_marital_premium(adjusted, input_data)
            adjusted = self._apply_age_adjustments(adjusted, input_data['age'])
            adjusted = self._apply_part_time_adjustment(adjusted, input_data)
            
            return {
                'annual': round(adjusted, 2),
                'monthly': round(adjusted / 12, 2),
                'currency': self._get_currency_symbol(input_data['native-country']),
                'hours_used': input_data['hours-per-week'],
                'age_group': self._get_age_group(input_data['age'])
            }
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

    def _apply_age_adjustments(self, salary, age):
        if age <= 25:
            salary *= 0.75
        elif 25 < age <= 50:
            salary *= (0.75 + 0.008 * (age - 25))
        elif age > 65:
            age_group = self._get_age_group(age)
            decay_factor = max(
                self.MIN_SALARY_RATIOS[age_group],
                1 - self.DECAY_RATES[age_group] * (age - 65)
            )
            salary *= decay_factor
        return salary

    def _get_age_group(self, age):
        if age >= 90:
            return 'centenarian'
        elif age >= 80:
            return 'elderly'
        elif age >= 65:
            return 'senior'
        elif age >= 50:
            return 'peak'
        elif age >= 25:
            return 'mid'
        return 'young'

    def _apply_part_time_adjustment(self, salary, input_data):
        full_time_hours = 40
        actual_hours = input_data['hours-per-week']
        
        if input_data['occupation'] in ('Healthcare', 'Executive', 'Prof-specialty'):
            full_time_hours = 50
        
        return salary * min(1.0, actual_hours / full_time_hours)

    def _apply_country_floor(self, prediction, input_data):
        """Ensure minimum salary requirements with proper line continuation"""
        country = input_data['native-country']
        occupation = input_data['occupation']
        age = input_data['age']
        hours = input_data['hours-per-week']
        
        # Use elderly minimums if applicable
        if age >= 65:
            floor = (self.ELDERLY_MINIMUMS.get(country, {}).get(occupation) or 
                     self.ELDERLY_MINIMUMS.get(country, {}).get('default', 0))
        else:
            floor = (self.country_floors.get(country, {}).get(occupation) or 
                     self.country_floors.get(country, {}).get('default', 0))
        
        # Apply occupation multiplier
        multiplier = self.OCCUPATION_FLOOR_MULTIPLIERS.get(
            occupation,
            self.OCCUPATION_FLOOR_MULTIPLIERS['default']
        )
        floor *= multiplier
        
        # Adjust for part-time (but ensure minimum 60%)
        if hours < 40:
            adjusted = floor * (hours / 40)
            floor = max(adjusted, 0.6 * floor)
        
        return max(prediction, floor)


    def _apply_occupation_baseline(self, prediction, input_data):
        baseline = self.occupation_baselines.get(input_data['occupation'], 0)
        age = input_data['age']
        
        if age < 50:
            experience_factor = min(1.0, 0.7 + 0.01 * (age - 18))
            baseline *= experience_factor
            
        return max(prediction, baseline * 0.85)

    def _apply_marital_premium(self, prediction, input_data):
        premium = self.marital_premiums.get(input_data['marital-status'], 1.0)
        
        if input_data['marital-status'] == 'Married-civ-spouse':
            if input_data['gender'] == 'Female':
                premium = max(1.0, premium * 0.97)
                
        return prediction * premium

    def _get_currency_symbol(self, country):
        symbols = {
            'USA': '$',
            'India': '₹',
            'Germany': '€',
            'UK': '£',
            'Japan': '¥',
            'default': '$'
        }
        return symbols.get(country, symbols['default'])

# Example usage
if __name__ == "__main__":
    predictor = EnhancedSalaryPredictor()
    test_input = {
        "age": 35,
        "workclass": "Private",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "gender": "Male",
        "native-country": "India",
        "hours-per-week": 40,
        "capital-gain": 0,
        "capital-loss": 0
    }
=======
# enhanced_predictor.py (Fixed Version)
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import joblib
from pathlib import Path

class EnhancedSalaryPredictor:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_info = None
        self.load_artifacts()
        
        # Configuration
        self.country_floors = {
            'USA': {'default': 15000, 'Handlers-cleaners': 18000, 'Craft-repair': 25000},
            'India': {
                'default': 200000,
                'Adm-clerical': 180000,
                'Handlers-cleaners': 250000,
                'Craft-repair': 250000,
                'Tech-support': 300000
            },
            'Germany': {'default': 12000}
        }
        
        self.occupation_baselines = {
            'Exec-managerial': 1500000,
            'Tech-support': 600000,
            'Adm-clerical': 360000,
            'Handlers-cleaners': 300000,
            'Craft-repair': 450000
        }
        
        self.marital_premiums = {
            'Married-civ-spouse': 1.08,
            'Never-married': 0.98,
            'Divorced': 1.0
        }
        
        self.MAX_AGE = 100
        self.MAX_HOURS = {
            'default': 40,
            'senior': 30,
            'elderly': 20,
            'centenarian': 10
        }

        self.ELDERLY_MINIMUMS = {
            'India': {
                'Handlers-cleaners': 220000,
                'Craft-repair': 200000,
                'default': 180000
            }
        }
        
        self.MIN_SALARY_RATIOS = {
            'senior': 0.75,
            'elderly': 0.65,
            'centenarian': 0.55
        }
        
        self.DECAY_RATES = {
            'senior': 0.008,
            'elderly': 0.012,
            'centenarian': 0.016
        }

        self.OCCUPATION_FLOOR_MULTIPLIERS = {
            'Handlers-cleaners': 1.4,
            'Craft-repair': 1.6,
            'Adm-clerical': 1.2,
            'Tech-support': 1.3,
            'default': 1.1
        }

    def load_artifacts(self):
        self.model = CatBoostRegressor()
        self.model.load_model(self.model_dir / "salary_model.cbm")
        self.feature_info = joblib.load(self.model_dir / "feature_info.joblib")

    def preprocess_input(self, input_data):
        self._validate_age(input_data['age'])
        input_data['hours-per-week'] = self._adjust_hours_for_age(
            input_data['hours-per-week'], 
            input_data['age']
        )
        
        df = pd.DataFrame([input_data])
        
        for feat in self.feature_info.get('derived_features', []):
            if feat == 'total_capital':
                df[feat] = df['capital-gain'] - df['capital-loss']
            elif feat.startswith('log_'):
                src_col = feat[4:]
                df[src_col] = np.log1p(df[src_col])
        
        df['capital-boost'] = np.log1p(df['capital-gain']) * 0.6
        df['experience-bonus'] = np.sqrt(df['age'] - 18) * 1200
        
        for feature in self.feature_info['feature_order'] + ['capital-boost', 'experience-bonus']:
            if feature not in df.columns:
                df[feature] = 0
                
        return df[self.feature_info['feature_order'] + ['capital-boost', 'experience-bonus']]

    def _validate_age(self, age):
        if not (18 <= age <= self.MAX_AGE):
            raise ValueError(
                f"Age {age} invalid. Must be 18-{self.MAX_AGE}. "
                f"Use 'force=True' to override (not recommended)."
            )

    def _adjust_hours_for_age(self, hours, age):
        if age >= 90:
            return min(hours, self.MAX_HOURS['centenarian'])
        elif age >= 80:
            return min(hours, self.MAX_HOURS['elderly'])
        elif age >= 65:
            return min(hours, self.MAX_HOURS['senior'])
        return min(hours, self.MAX_HOURS['default'])

    def predict(self, input_data, force=False):
        try:
            if not force:
                self._validate_age(input_data['age'])
            
            input_df = self.preprocess_input(input_data)
            pool = Pool(input_df, cat_features=self.feature_info['categorical_cols'])
            raw_pred = np.expm1(self.model.predict(pool)[0])
            
            adjusted = self._apply_country_floor(raw_pred, input_data)
            adjusted = self._apply_occupation_baseline(adjusted, input_data)
            adjusted = self._apply_marital_premium(adjusted, input_data)
            adjusted = self._apply_age_adjustments(adjusted, input_data['age'])
            adjusted = self._apply_part_time_adjustment(adjusted, input_data)
            
            return {
                'annual': round(adjusted, 2),
                'monthly': round(adjusted / 12, 2),
                'currency': self._get_currency_symbol(input_data['native-country']),
                'hours_used': input_data['hours-per-week'],
                'age_group': self._get_age_group(input_data['age'])
            }
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

    def _apply_age_adjustments(self, salary, age):
        if age <= 25:
            salary *= 0.75
        elif 25 < age <= 50:
            salary *= (0.75 + 0.008 * (age - 25))
        elif age > 65:
            age_group = self._get_age_group(age)
            decay_factor = max(
                self.MIN_SALARY_RATIOS[age_group],
                1 - self.DECAY_RATES[age_group] * (age - 65)
            )
            salary *= decay_factor
        return salary

    def _get_age_group(self, age):
        if age >= 90:
            return 'centenarian'
        elif age >= 80:
            return 'elderly'
        elif age >= 65:
            return 'senior'
        elif age >= 50:
            return 'peak'
        elif age >= 25:
            return 'mid'
        return 'young'

    def _apply_part_time_adjustment(self, salary, input_data):
        full_time_hours = 40
        actual_hours = input_data['hours-per-week']
        
        if input_data['occupation'] in ('Healthcare', 'Executive', 'Prof-specialty'):
            full_time_hours = 50
        
        return salary * min(1.0, actual_hours / full_time_hours)

    def _apply_country_floor(self, prediction, input_data):
        """Ensure minimum salary requirements with proper line continuation"""
        country = input_data['native-country']
        occupation = input_data['occupation']
        age = input_data['age']
        hours = input_data['hours-per-week']
        
        # Use elderly minimums if applicable
        if age >= 65:
            floor = (self.ELDERLY_MINIMUMS.get(country, {}).get(occupation) or 
                     self.ELDERLY_MINIMUMS.get(country, {}).get('default', 0))
        else:
            floor = (self.country_floors.get(country, {}).get(occupation) or 
                     self.country_floors.get(country, {}).get('default', 0))
        
        # Apply occupation multiplier
        multiplier = self.OCCUPATION_FLOOR_MULTIPLIERS.get(
            occupation,
            self.OCCUPATION_FLOOR_MULTIPLIERS['default']
        )
        floor *= multiplier
        
        # Adjust for part-time (but ensure minimum 60%)
        if hours < 40:
            adjusted = floor * (hours / 40)
            floor = max(adjusted, 0.6 * floor)
        
        return max(prediction, floor)


    def _apply_occupation_baseline(self, prediction, input_data):
        baseline = self.occupation_baselines.get(input_data['occupation'], 0)
        age = input_data['age']
        
        if age < 50:
            experience_factor = min(1.0, 0.7 + 0.01 * (age - 18))
            baseline *= experience_factor
            
        return max(prediction, baseline * 0.85)

    def _apply_marital_premium(self, prediction, input_data):
        premium = self.marital_premiums.get(input_data['marital-status'], 1.0)
        
        if input_data['marital-status'] == 'Married-civ-spouse':
            if input_data['gender'] == 'Female':
                premium = max(1.0, premium * 0.97)
                
        return prediction * premium

    def _get_currency_symbol(self, country):
        symbols = {
            'USA': '$',
            'India': '₹',
            'Germany': '€',
            'UK': '£',
            'Japan': '¥',
            'default': '$'
        }
        return symbols.get(country, symbols['default'])

# Example usage
if __name__ == "__main__":
    predictor = EnhancedSalaryPredictor()
    test_input = {
        "age": 35,
        "workclass": "Private",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "gender": "Male",
        "native-country": "India",
        "hours-per-week": 40,
        "capital-gain": 0,
        "capital-loss": 0
    }
>>>>>>> d3f8614 (Initial commit with CareerWorth app)
    print(f"Predicted Salary: {predictor.predict(test_input)}")