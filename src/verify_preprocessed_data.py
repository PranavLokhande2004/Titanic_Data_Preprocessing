import pandas as pd

# Load the preprocessed dataset
df = pd.read_csv("data/titanic_preprocessed.csv")

print("--- Preprocessed Dataset Verification ---")

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())