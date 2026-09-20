import pandas as pd

# Load the raw Titanic dataset
df = pd.read_csv("data/titanic_raw.csv")

print("--- Original Dataset Shape ---")
print(df.shape)

# --------------------------------------------------
# 1. Remove duplicate rows
# --------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\n--- Duplicate Rows ---")
print("Duplicate rows found:", duplicate_count)

df = df.drop_duplicates()

# --------------------------------------------------
# 2. Handle missing Age values
# --------------------------------------------------

age_missing = df["Age"].isnull().sum()

print("\n--- Missing Age ---")
print("Missing Age values:", age_missing)

age_median = df["Age"].median()

df["Age"] = df["Age"].fillna(age_median)

print("Age missing values after cleaning:",
      df["Age"].isnull().sum())

# --------------------------------------------------
# 3. Handle missing Embarked values
# --------------------------------------------------

embarked_missing = df["Embarked"].isnull().sum()

print("\n--- Missing Embarked ---")
print("Missing Embarked values:", embarked_missing)

embarked_mode = df["Embarked"].mode()[0]

df["Embarked"] = df["Embarked"].fillna(embarked_mode)

print("Embarked missing values after cleaning:",
      df["Embarked"].isnull().sum())

# --------------------------------------------------
# 4. Handle missing Cabin values
# --------------------------------------------------

cabin_missing = df["Cabin"].isnull().sum()

print("\n--- Missing Cabin ---")
print("Missing Cabin values:", cabin_missing)

df["Cabin"] = df["Cabin"].fillna("Unknown")

print("Cabin missing values after cleaning:",
      df["Cabin"].isnull().sum())

# --------------------------------------------------
# 5. Flag Fare outliers
# --------------------------------------------------

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

upper_bound = Q3 + 1.5 * IQR

df["Fare_Outlier"] = df["Fare"] > upper_bound

print("\n--- Fare Outliers ---")
print("Upper Bound:", upper_bound)
print("Fare outliers:", df["Fare_Outlier"].sum())

# --------------------------------------------------
# 6. Check remaining missing values
# --------------------------------------------------

print("\n--- Remaining Missing Values ---")
print(df.isnull().sum())

# --------------------------------------------------
# 7. Final dataset shape
# --------------------------------------------------

print("\n--- Cleaned Dataset Shape ---")
print(df.shape)



# Save the cleaned dataset
output_path = "data/titanic_cleaned.csv"

df.to_csv(output_path, index=False)

print("\n--- Cleaned Dataset Saved ---")
print("File saved at:", output_path)