import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("data/titanic_cleaned.csv")

# Create FamilySize
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Create FamilySize distribution
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="FamilySize"
)

plt.title("Distribution of Passenger Family Size")
plt.xlabel("Family Size")
plt.ylabel("Number of Passengers")

plt.tight_layout()

# Save the chart
plt.savefig("outputs/family_size_distribution.png")

plt.show()