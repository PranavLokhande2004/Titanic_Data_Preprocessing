import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("Week 3 - Unsupervised Learning and Clustering")
print("=" * 50)

# Load the cleaned Titanic dataset
df = pd.read_csv("data/titanic_cleaned.csv")

# Create FamilySize
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# Select features for clustering
features = ["Pclass", "Age", "Fare", "FamilySize"]

X = df[features]

print("\nFeatures selected for clustering:")
print(features)

print("\nClustering data preview:")
print(X.head())

print("\nClustering data shape:")
print(X.shape)

print("\nMissing values in clustering data:")
print(X.isnull().sum())

# Standardize the features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeatures standardized successfully!")

print("\nFirst 5 standardized rows:")
print(X_scaled[:5])

print("\nStandardized data shape:")
print(X_scaled.shape)


# Elbow Method to find the optimal number of clusters

inertia = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

print("\nElbow Method Results:")

for k, value in zip(range(2, 9), inertia):
    print(f"K = {k}, Inertia = {value:.2f}")

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(2, 9), inertia, marker="o")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(2, 9))
plt.grid(True)
plt.tight_layout()

plt.savefig("week3_outputs/elbow_method.png")

plt.show()


# Apply K-Means clustering with 4 clusters

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means clustering completed successfully!")

print("\nCluster counts:")
print(df["Cluster"].value_counts().sort_index())


# Analyze characteristics of each cluster

cluster_summary = df.groupby("Cluster")[features].mean()

print("\nCluster Characteristics:")
print(cluster_summary)

print("\nNumber of passengers in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# Visualize the clusters

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Age",
    y="Fare",
    hue="Cluster",
    palette="Set1",
    s=60
)

plt.title("Titanic Passenger Clusters")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(title="Cluster")
plt.tight_layout()

plt.savefig("week3_outputs/kmeans_clusters.png")

plt.show()


# Compare clusters with survival rate

cluster_survival = df.groupby("Cluster")["Survived"].mean() * 100

print("\nSurvival Rate by Cluster:")
print(cluster_survival.round(2))

# Visualize survival rate by cluster

plt.figure(figsize=(8, 5))

sns.barplot(
    x=cluster_survival.index,
    y=cluster_survival.values
)

plt.title("Survival Rate by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Survival Rate (%)")
plt.tight_layout()

plt.savefig("week3_outputs/survival_by_cluster.png")

plt.show()



# Save cluster summary for further analysis

cluster_summary = df.groupby("Cluster")[features].mean()
cluster_summary["Passenger_Count"] = df["Cluster"].value_counts().sort_index()
cluster_summary["Survival_Rate"] = (
    df.groupby("Cluster")["Survived"].mean() * 100
)

cluster_summary = cluster_summary.round(2)

cluster_summary.to_csv("week3_outputs/cluster_summary.csv")

print("\nCluster summary saved successfully!")
print("\nFinal Cluster Summary:")
print(cluster_summary)

# Visualize average feature values for each cluster

cluster_features = df.groupby("Cluster")[features].mean()

plt.figure(figsize=(10, 6))

cluster_features.plot(kind="bar", figsize=(10, 6))

plt.title("Average Feature Values by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.legend(title="Features")
plt.tight_layout()

plt.savefig("week3_outputs/cluster_feature_comparison.png")

plt.show()
