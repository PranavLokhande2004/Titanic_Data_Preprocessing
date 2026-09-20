import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/titanic_cleaned.csv")

print("--- Cleaned Dataset ---")

# Show first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Show dataset shape
print("\nDataset Shape:")
print(df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Show column names
print("\nColumn Names:")
print(df.columns.tolist())