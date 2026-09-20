import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/titanic_cleaned.csv")

print("--- Before Preprocessing ---")
print(df.shape)

# Create FamilySize feature
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

print("\n--- FamilySize Created ---")
print(df[["SibSp", "Parch", "FamilySize"]].head(10))

# Display FamilySize statistics
print("\n--- FamilySize Statistics ---")
print(df["FamilySize"].describe())

# Create IsAlone feature
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

print("\n--- IsAlone Created ---")
print(df[["FamilySize", "IsAlone"]].head(10))

# Count passengers travelling alone and with family
print("\n--- IsAlone Counts ---")
print(df["IsAlone"].value_counts())


# Convert categorical columns into numerical columns
df = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True,
    dtype=int
)

print("\n--- After One-Hot Encoding ---")
print(df.head())

print("\n--- New Column Names ---")
print(df.columns.tolist())

# Save the preprocessed dataset
output_path = "data/titanic_preprocessed.csv"

df.to_csv(output_path, index=False)

print("\n--- Preprocessed Dataset Saved ---")
print("File saved at:", output_path)
print("Final Shape:", df.shape)