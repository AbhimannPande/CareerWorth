<<<<<<< HEAD
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('employee_salaries_custom.csv')  # Replace with your dataset path

# Basic info
print("Dataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSample data:")
print(df.head())

# Unique values for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}':")
    print(df[col].value_counts())

# Histograms for numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_cols].hist(figsize=(12, 8))
plt.tight_layout()
=======
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('employee_salaries_custom.csv')  # Replace with your dataset path

# Basic info
print("Dataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSample data:")
print(df.head())

# Unique values for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}':")
    print(df[col].value_counts())

# Histograms for numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_cols].hist(figsize=(12, 8))
plt.tight_layout()
>>>>>>> d3f8614 (Initial commit with CareerWorth app)
plt.show()