"""
==========================================================
Iris Flower Classification
QSkill AI/ML Internship Task

Author: Anik Chaudhary
==========================================================
"""

import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(__file__), ".matplotlib"),
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["Species"] = iris.target
df["Species"] = df["Species"].map(
    {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica",
    }
)

print("=" * 60)
print("First Five Rows")
print("=" * 60)
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nClass Distribution")
print(df["Species"].value_counts())

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

sns.set(style="whitegrid")

# Pairplot

pair = sns.pairplot(
    df,
    hue="Species",
    corner=True
)

pair.fig.suptitle(
    "Iris Pairplot",
    y=1.02
)

plt.savefig("pairplot.png")
plt.close()

# Histograms

df.hist(figsize=(10, 8))
plt.tight_layout()
plt.savefig("histograms.png")
plt.close()

# ---------------------------------------------------------
# Prepare Data
# ---------------------------------------------------------

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Standardize Features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------
# Train Models
# ---------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

results = {}

print("\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}\n")

    print("Classification Report")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names,
        )
    )

# ---------------------------------------------------------
# Best Model
# ---------------------------------------------------------

best_model_name = max(results, key=results.get)

print("=" * 60)
print("Best Model:", best_model_name)
print("=" * 60)

best_model = models[best_model_name]

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.close()

print("\nFinished Successfully!")
