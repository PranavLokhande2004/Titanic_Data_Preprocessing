# ==========================================
# WEEK 6: INTEGRATIVE CAPSTONE PROJECT
# Titanic Survival Prediction
# ==========================================

# Step 1: Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Step 2: Define project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "titanic_cleaned.csv"

OUTPUT_PATH = Path(__file__).resolve().parent / "outputs"

# Create output directory if it does not exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Step 3: Load the Titanic dataset

df = pd.read_csv(DATA_PATH)

# Step 4: Display basic dataset information

print("\n========== TITANIC CAPSTONE PROJECT ==========")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# STEP 5: DATASET ANALYSIS
# ==========================================

print("\n========== DATASET ANALYSIS ==========")

# Total number of passengers
total_passengers = len(df)

# Count survived and not survived
survival_counts = df["Survived"].value_counts().sort_index()

# Calculate survival percentage
survival_percentage = df["Survived"].value_counts(
    normalize=True
).sort_index() * 100

print("\nTotal Passengers:", total_passengers)

print("\nSurvival Count:")
print(survival_counts)

print("\nSurvival Percentage:")
print(survival_percentage.round(2))

# Gender-wise survival analysis
print("\nGender-wise Survival Rate:")

gender_survival = df.groupby("Sex")["Survived"].mean() * 100

print(gender_survival.round(2))

# Passenger class-wise survival analysis
print("\nPassenger Class-wise Survival Rate:")

class_survival = df.groupby("Pclass")["Survived"].mean() * 100

print(class_survival.round(2))

# Save analysis results
survival_counts.to_csv(
    OUTPUT_PATH / "survival_counts.csv"
)

gender_survival.to_csv(
    OUTPUT_PATH / "gender_survival.csv"
)

class_survival.to_csv(
    OUTPUT_PATH / "class_survival.csv"
)

print("\nAnalysis results saved successfully!")

# ==========================================
# STEP 6: DATA VISUALIZATION
# ==========================================

print("\n========== CREATING VISUALIZATIONS ==========")

# 1. Overall survival distribution
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Survived")

plt.title("Titanic Passenger Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.savefig(OUTPUT_PATH / "capstone_survival_distribution.png")
plt.close()


# 2. Survival by gender
plt.figure(figsize=(7, 5))

sns.barplot(data=df, x="Sex", y="Survived")

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

plt.savefig(OUTPUT_PATH / "capstone_survival_by_gender.png")
plt.close()


# 3. Survival by passenger class
plt.figure(figsize=(7, 5))

sns.barplot(data=df, x="Pclass", y="Survived")

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.savefig(OUTPUT_PATH / "capstone_survival_by_class.png")
plt.close()


# 4. Age distribution
plt.figure(figsize=(7, 5))

sns.histplot(data=df, x="Age", bins=30, kde=True)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.savefig(OUTPUT_PATH / "capstone_age_distribution.png")
plt.close()


print("\nAll visualizations created successfully!")

# ==========================================
# STEP 7: FINAL MODEL COMPARISON
# ==========================================

print("\n========== FINAL MODEL COMPARISON ==========")

# Load existing model comparison results
model_results_path = (
    PROJECT_ROOT / "week5_outputs" / "model_comparison.csv"
)

model_results = pd.read_csv(model_results_path)

# Display comparison table
print("\nComparison of Machine Learning Models:")
print(model_results.to_string(index=False))

# Save consolidated results in Week 6
model_results.to_csv(
    OUTPUT_PATH / "final_model_comparison.csv",
    index=False
)

# Create model comparison visualization
model_metrics = model_results.set_index("Metric")

model_metrics.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Comparison of Machine Learning Models")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(title="Models")
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "final_model_comparison.png"
)

plt.close()

print("\nFinal model comparison saved successfully!")


# ==========================================
# STEP 8: CLUSTERING ANALYSIS
# ==========================================

print("\n========== CLUSTERING ANALYSIS ==========")

# Load Week 3 clustering results
cluster_path = (
    PROJECT_ROOT / "week3_outputs" / "cluster_summary.csv"
)

cluster_results = pd.read_csv(cluster_path)

# Display cluster summary
print("\nPassenger Cluster Summary:")
print(cluster_results.to_string(index=False))

# Save consolidated clustering results
cluster_results.to_csv(
    OUTPUT_PATH / "final_cluster_summary.csv",
    index=False
)

# Create cluster survival visualization
plt.figure(figsize=(8, 5))

sns.barplot(
    data=cluster_results,
    x="Cluster",
    y="Survival_Rate"
)

plt.title("Survival Rate Across Passenger Clusters")
plt.xlabel("Cluster")
plt.ylabel("Survival Rate (%)")

plt.savefig(
    OUTPUT_PATH / "final_survival_by_cluster.png"
)

plt.close()

print("\nClustering analysis integrated successfully!")


# ==========================================
# STEP 9: PROJECT INSIGHTS AND RECOMMENDATIONS
# ==========================================

print("\n========== PROJECT INSIGHTS ==========")

# Overall survival rate
overall_survival = df["Survived"].mean() * 100

# Gender with higher survival rate
gender_rates = df.groupby("Sex")["Survived"].mean() * 100
highest_gender = gender_rates.idxmax()

# Passenger class with higher survival rate
class_rates = df.groupby("Pclass")["Survived"].mean() * 100
highest_class = class_rates.idxmax()

# Best model according to accuracy
accuracy_row = model_results[
    model_results["Metric"] == "Accuracy"
].iloc[0]

best_model = accuracy_row.drop("Metric").astype(float).idxmax()
best_accuracy = accuracy_row[best_model] * 100

# Cluster with highest survival rate
best_cluster = cluster_results.loc[
    cluster_results["Survival_Rate"].idxmax()
]

# Display insights
print(f"\nOverall Survival Rate: {overall_survival:.2f}%")

print(
    f"Gender with Highest Survival Rate: {highest_gender}"
)

print(
    f"Passenger Class with Highest Survival Rate: {highest_class}"
)

print(
    f"Best Model by Accuracy: {best_model}"
)

print(
    f"Best Model Accuracy: {best_accuracy:.2f}%"
)

print(
    f"Cluster with Highest Survival Rate: "
    f"{int(best_cluster['Cluster'])}"
)

print(
    f"Highest Cluster Survival Rate: "
    f"{best_cluster['Survival_Rate']:.2f}%"
)

# Save insights to a text file
insights = f"""
TITANIC CAPSTONE PROJECT - FINAL INSIGHTS

1. Overall survival rate: {overall_survival:.2f}%

2. Gender with highest survival rate: {highest_gender}

3. Passenger class with highest survival rate: {highest_class}

4. Best model by accuracy: {best_model}

5. Best model accuracy: {best_accuracy:.2f}%

6. Cluster with highest survival rate:
   Cluster {int(best_cluster['Cluster'])}

7. Highest cluster survival rate:
   {best_cluster['Survival_Rate']:.2f}%
"""

with open(
    OUTPUT_PATH / "final_project_insights.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(insights)

print("\nProject insights saved successfully!")


# ==========================================
# STEP 10: LIMITATIONS AND FUTURE IMPROVEMENTS
# ==========================================

print("\n========== LIMITATIONS AND FUTURE IMPROVEMENTS ==========")

limitations = [
    "The Titanic dataset contains only 891 passenger records.",
    "Cabin information has a large number of missing values.",
    "The dataset represents one historical disaster and may not generalize to other situations.",
    "The neural network may not outperform traditional machine learning models on small tabular data.",
    "Model performance depends on the selected features and train-test split."
]

future_improvements = [
    "Use additional passenger-related features if reliable data is available.",
    "Apply hyperparameter tuning to improve model performance.",
    "Experiment with additional classification algorithms.",
    "Use cross-validation for more reliable model evaluation.",
    "Test the workflow on larger and different datasets."
]

print("\nProject Limitations:")
for number, item in enumerate(limitations, start=1):
    print(f"{number}. {item}")

print("\nFuture Improvements:")
for number, item in enumerate(future_improvements, start=1):
    print(f"{number}. {item}")

# Save limitations and improvements
with open(
    OUTPUT_PATH / "limitations_and_future_improvements.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("PROJECT LIMITATIONS\n\n")

    for item in limitations:
        file.write("- " + item + "\n")

    file.write("\nFUTURE IMPROVEMENTS\n\n")

    for item in future_improvements:
        file.write("- " + item + "\n")

print("\nLimitations and future improvements saved successfully!")