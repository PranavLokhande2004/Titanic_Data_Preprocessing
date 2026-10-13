
# Week 5: Deep Learning Application in Data Science
# Project: Titanic Survival Prediction Using a Neural Network

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("Libraries imported successfully!")



# Step 5: Load the Titanic dataset

df = pd.read_csv("../data/titanic_cleaned.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nAvailable columns:")
print(df.columns.tolist())

# Separate the target and input features
X = df.drop(columns=["Survived", "PassengerId"], errors="ignore")
y = df["Survived"]

print("\nInput features shape:", X.shape)
print("Target shape:", y.shape)
print("\nSurvival distribution:")
print(y.value_counts())


# Step 6: Prepare numerical and categorical features

# Identify the feature types
numerical_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

# Prepare numerical data
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Prepare categorical data
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

print("\nPreprocessing pipeline created successfully!")
print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)


# Step 7: Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining and testing split completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Fit preprocessing on training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert sparse matrices to dense arrays if needed
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

# Convert data into TensorFlow-compatible arrays
X_train_processed = X_train_processed.astype("float32")
X_test_processed = X_test_processed.astype("float32")

y_train = y_train.to_numpy().astype("float32")
y_test = y_test.to_numpy().astype("float32")

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)
print("Data is ready for the neural network!")



# Step 8: Build the Artificial Neural Network (ANN)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# Create the neural network architecture
model = Sequential([
    Input(shape=(X_train_processed.shape[1],)),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(16, activation="relu"),
    Dropout(0.2),

    Dense(1, activation="sigmoid")
])

# Configure the model for binary classification
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display the network architecture
print("\nNeural network created successfully!")
model.summary()



# Step 9: Train the neural network

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

print("\nTraining the neural network...")

history = model.fit(
    X_train_processed,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

print("\nTraining completed!")
print("Epochs completed:", len(history.history["loss"]))



# Step 10: Evaluate the trained neural network

from sklearn.metrics import confusion_matrix

# Evaluate loss and accuracy on unseen test data
test_loss, test_accuracy = model.evaluate(
    X_test_processed,
    y_test,
    verbose=0
)

# Predict survival probabilities
y_prob = model.predict(X_test_processed, verbose=0).ravel()

# Convert probabilities into classes
y_pred = (y_prob >= 0.5).astype(int)

# Calculate evaluation metrics
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== NEURAL NETWORK EVALUATION =====")
print(f"Test Loss:  {test_loss:.4f}")
print(f"Accuracy:   {test_accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-Score:   {f1:.4f}")
print(f"ROC-AUC:    {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# Step 11: Visualize training and evaluation results

import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create an output folder
os.makedirs("../week5_outputs", exist_ok=True)

# Plot training and validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../week5_outputs/training_accuracy.png")
plt.close()

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../week5_outputs/training_loss.png")
plt.close()

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Did Not Survive", "Survived"],
    yticklabels=["Did Not Survive", "Survived"]
)
plt.title("Neural Network Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("../week5_outputs/confusion_matrix.png")
plt.close()

print("\nGraphs saved successfully in week5_outputs!")



# Step 13: Save neural network evaluation metrics

results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Neural Network": [
        test_accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

os.makedirs("../week5_outputs", exist_ok=True)

results.to_csv(
    "../week5_outputs/neural_network_metrics.csv",
    index=False
)

print("\nNeural network metrics saved successfully!")
print(results.to_string(index=False))



# Step 17: Compare Week 4 models with the neural network

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Logistic Regression": [0.804469, 0.793103, 0.666667, 0.724409, 0.844532],
    "Random Forest": [0.826816, 0.816667, 0.710145, 0.759690, 0.831423],
    "Neural Network": [
        test_accuracy, precision, recall, f1, roc_auc
    ]
})

comparison.to_csv(
    "../week5_outputs/model_comparison.csv",
    index=False
)

print("\n===== THREE-MODEL COMPARISON =====")
print(comparison.round(4).to_string(index=False))

# Create a comparison chart
comparison_plot = comparison.set_index("Metric")

comparison_plot.plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1)
)

plt.title("Machine Learning vs Deep Learning")
plt.ylabel("Score")
plt.xlabel("Evaluation Metric")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../week5_outputs/model_comparison.png")
plt.close()

print("\nComparison CSV and chart saved successfully!")
