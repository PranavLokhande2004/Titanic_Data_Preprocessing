import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned Titanic dataset
df = pd.read_csv("data/titanic_cleaned.csv")

# Create FamilySize for EDA
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

print("Titanic Dataset - Week 2 EDA")
print("=" * 40)

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display data types
print("\nData Types:")
print(df.dtypes)

# Basic statistical analysis
print("\n" + "=" * 40)
print("Statistical Summary")
print("=" * 40)

print(df.describe())

# Analyze important categorical features
print("\n" + "=" * 40)
print("Feature Analysis")
print("=" * 40)

print("\nPassenger Class:")
print(df["Pclass"].value_counts())

print("\nGender:")
print(df["Sex"].value_counts())

print("\nEmbarked:")
print(df["Embarked"].value_counts())

print("\nSurvival:")
print(df["Survived"].value_counts())

# Visualization 1: Survival Distribution
plt.figure(figsize=(6, 4))

sns.countplot(x="Survived", data=df)

plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig("week2_outputs/survival_distribution.png")
plt.show()

# Visualization 2: Survival by Gender
plt.figure(figsize=(6, 4))

sns.countplot(x="Sex", hue="Survived", data=df)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.legend(title="Survived", labels=["No", "Yes"])

plt.tight_layout()
plt.savefig("week2_outputs/survival_by_gender.png")
plt.show()

# Visualization 3: Survival by Passenger Class
plt.figure(figsize=(6, 4))

sns.countplot(x="Pclass", hue="Survived", data=df)

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.legend(title="Survived", labels=["No", "Yes"])

plt.tight_layout()
plt.savefig("week2_outputs/survival_by_class.png")
plt.show()


# Visualization 4: Age Distribution
plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="Age", bins=20, kde=True)

plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig("week2_outputs/age_distribution.png")
plt.show()


# Visualization 5: Fare Distribution and Outliers
plt.figure(figsize=(8, 5))

sns.boxplot(x=df["Fare"])

plt.title("Fare Distribution and Outliers")
plt.xlabel("Fare")

plt.tight_layout()
plt.savefig("week2_outputs/fare_outliers.png")
plt.show()

# Visualization 6: Correlation Heatmap
plt.figure(figsize=(10, 7))

numeric_df = df.select_dtypes(include="number")
correlation_matrix = numeric_df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("week2_outputs/correlation_heatmap.png")
plt.show()


# Visualization 7: Family Size Distribution
plt.figure(figsize=(8, 5))

sns.countplot(x="FamilySize", data=df)

plt.title("Family Size Distribution")
plt.xlabel("Family Size")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig("week2_outputs/family_size_distribution.png")
plt.show()

# Grouped Analysis:

# ==========================================
# All Survival Rate Analysis
# ==========================================

print("\n" + "=" * 50)
print("ALL SURVIVAL RATE ANALYSIS")
print("=" * 50)

# 1. Overall Survival Rate
overall_survival = df["Survived"].mean() * 100
print(f"\nOverall Survival Rate: {overall_survival:.2f}%")

# 2. Survival Rate by Gender
print("\nSurvival Rate by Gender:")
print((df.groupby("Sex")["Survived"].mean() * 100).round(2))

# 3. Survival Rate by Passenger Class
print("\nSurvival Rate by Passenger Class:")
print((df.groupby("Pclass")["Survived"].mean() * 100).round(2))

# 4. Survival Rate by Family Size
print("\nSurvival Rate by Family Size:")
print((df.groupby("FamilySize")["Survived"].mean() * 100).round(2))

# 5. Create Age Groups
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teenager", "Young Adult", "Adult", "Senior"]
)

# 6. Survival Rate by Age Group
print("\nSurvival Rate by Age Group:")
print(
    (df.groupby("AgeGroup", observed=False)["Survived"].mean() * 100)
    .round(2)
)


# Visualization 8: Survival Rate by Age Group
age_survival = df.groupby("AgeGroup", observed=False)["Survived"].mean() * 100

plt.figure(figsize=(8, 5))

sns.barplot(
    x=age_survival.index,
    y=age_survival.values
)

plt.title("Survival Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("week2_outputs/survival_by_age_group.png")
plt.show()

# ==========================================
# Save EDA Summary
# ==========================================

summary = {
    "Total Passengers": len(df),
    "Overall Survival Rate (%)": round(df["Survived"].mean() * 100, 2),
    "Male Survival Rate (%)": round(
        df[df["Sex"] == "male"]["Survived"].mean() * 100, 2
    ),
    "Female Survival Rate (%)": round(
        df[df["Sex"] == "female"]["Survived"].mean() * 100, 2
    ),
    "1st Class Survival Rate (%)": round(
        df[df["Pclass"] == 1]["Survived"].mean() * 100, 2
    ),
    "2nd Class Survival Rate (%)": round(
        df[df["Pclass"] == 2]["Survived"].mean() * 100, 2
    ),
    "3rd Class Survival Rate (%)": round(
        df[df["Pclass"] == 3]["Survived"].mean() * 100, 2
    )
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["Metric", "Value"]
)

summary_df.to_csv(
    "week2_outputs/eda_summary.csv",
    index=False
)

print("\n" + "=" * 50)
print("EDA SUMMARY")
print("=" * 50)
print(summary_df)
print("\nEDA summary saved successfully!")

# ==========================================
# Correlation Analysis
# ==========================================

print("\n" + "=" * 50)
print("Correlation with Survival")
print("=" * 50)

numeric_df = df.select_dtypes(include="number")

survival_correlation = (
    numeric_df.corr()["Survived"]
    .sort_values(ascending=False)
)

print(survival_correlation)



# Save key EDA findings
findings = {
    "Finding": [
        "Overall survival rate",
        "Female survival rate",
        "Male survival rate",
        "1st Class survival rate",
        "2nd Class survival rate",
        "3rd Class survival rate",
        "Highest family-size survival rate",
        "Lowest age-group survival rate",
        "Strongest correlation with survival",
        "Potential Fare outliers"
    ],
    "Value": [
        "38.38%",
        "74.20%",
        "18.89%",
        "62.96%",
        "47.28%",
        "24.24%",
        "72.41% (FamilySize 4)",
        "22.73% (Senior)",
        "Pclass (-0.3385)",
        "116"
    ]
}

findings_df = pd.DataFrame(findings)

findings_df.to_csv(
    "week2_outputs/eda_findings.csv",
    index=False
)

print("\nEDA findings saved successfully!")