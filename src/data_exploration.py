import pandas as pd

# Load the Titanic dataset
df = pd.read_csv("data/titanic_raw.csv")

# Display first 5 rows
print("\n--- First 5 Rows ---")
print(df.head())

# Display number of rows and columns
print("\n--- Dataset Shape ---")
print(df.shape)

# Display column names
print("\n--- Column Names ---")
print(df.columns)

# Display information about the dataset
print("\n--- Dataset Information ---")
print(df.info())

# Display statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())

# Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Calculate missing-value percentage
print("\n--- Missing Value Percentage ---")

missing_percentage = (df.isnull().sum() / len(df)) * 100

print(missing_percentage)

# Check duplicate rows
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# Check invalid Age values
print("\n--- Invalid Age Values ---")
print(df[df["Age"] < 0])

# Check invalid Fare values
print("\n--- Invalid Fare Values ---")
print(df[df["Fare"] < 0])

# Check unique values in Sex
print("\n--- Unique Sex Values ---")
print(df["Sex"].unique())

# Check unique values in Embarked
print("\n--- Unique Embarked Values ---")
print(df["Embarked"].unique())

# Check unique values in Pclass
print("\n--- Unique Pclass Values ---")
print(df["Pclass"].unique())

# Check unique values in Survived
print("\n--- Unique Survived Values ---")
print(df["Survived"].unique())


# Detect Fare outliers using IQR method
print("\n--- Fare Outlier Analysis ---")

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

fare_outliers = df[
    (df["Fare"] < lower_bound) |
    (df["Fare"] > upper_bound)
]

print("Number of Fare Outliers:", len(fare_outliers))
print("Minimum Outlier Fare:", fare_outliers["Fare"].min())
print("Maximum Outlier Fare:", fare_outliers["Fare"].max())