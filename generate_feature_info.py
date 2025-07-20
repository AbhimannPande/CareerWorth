<<<<<<< HEAD
import joblib

# Load existing feature_info
feature_info = joblib.load('feature_info.joblib')

# Add numeric_cols manually
feature_info['numeric_cols'] = [
    'age', 'hours-per-week', 'capital-gain', 'capital-loss', 'total_capital'
]

# Save back with numeric_cols included
joblib.dump(feature_info, 'feature_info.joblib')

print("✅ feature_info.joblib updated with numeric_cols!")
print(feature_info)
=======
import joblib

# Load existing feature_info
feature_info = joblib.load('feature_info.joblib')

# Add numeric_cols manually
feature_info['numeric_cols'] = [
    'age', 'hours-per-week', 'capital-gain', 'capital-loss', 'total_capital'
]

# Save back with numeric_cols included
joblib.dump(feature_info, 'feature_info.joblib')

print("✅ feature_info.joblib updated with numeric_cols!")
print(feature_info)
>>>>>>> d3f8614 (Initial commit with CareerWorth app)
