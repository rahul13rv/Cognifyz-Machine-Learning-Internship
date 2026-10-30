# ============================================================
# CUISINE CLASSIFICATION - MACHINE LEARNING MODEL
# ============================================================

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("Dataset.csv", encoding="utf-8-sig")

print("Dataset shape:", df.shape)
print("\nColumns:", list(df.columns))

# ------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------

# Drop duplicates by restaurant ID
df = df.drop_duplicates(subset="Restaurant ID").copy()

# Convert numeric columns safely
for col in ["Average Cost for two", "Aggregate rating", "Votes", "Price range"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values
df["Cuisines"] = df["Cuisines"].fillna("Unknown")
df["City"] = df["City"].fillna("Unknown")
df["Locality"] = df["Locality"].fillna("Unknown")
df["Has Table booking"] = df["Has Table booking"].fillna("No")
df["Has Online delivery"] = df["Has Online delivery"].fillna("No")
df["Average Cost for two"] = df["Average Cost for two"].fillna(
    df["Average Cost for two"].median()
)
df["Aggregate rating"] = df["Aggregate rating"].fillna(
    df["Aggregate rating"].median()
)
df["Votes"] = df["Votes"].fillna(0)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 3. TARGET COLUMN: PRIMARY CUISINE
# ------------------------------------------------------------

# Each row's Cuisines looks like "Japanese, Sushi"
# We take the FIRST cuisine as the primary label.
df["Primary Cuisine"] = (
    df["Cuisines"]
    .astype(str)
    .apply(lambda x: x.split(",")[0].strip())
)

print("\nTop 20 primary cuisines:")
print(df["Primary Cuisine"].value_counts().head(20))

# ------------------------------------------------------------
# 4. KEEP ONLY TOP N CUISINES
# ------------------------------------------------------------

TOP_N = 12   # you can try 10, 15, or 20

top_cuisines = df["Primary Cuisine"].value_counts().head(TOP_N).index.tolist()

df_model = df[df["Primary Cuisine"].isin(top_cuisines)].copy()

print(f"\nKept {len(df_model)} rows across {TOP_N} cuisines.")
print("Class distribution:")
print(df_model["Primary Cuisine"].value_counts())

# ------------------------------------------------------------
# 5. FEATURE SELECTION
# ------------------------------------------------------------

# Numeric features
num_features = [
    "Average Cost for two",
    "Aggregate rating",
    "Votes",
    "Price range"
]

# Categorical features
cat_features = [
    "City",
    "Has Table booking",
    "Has Online delivery",
    "Country Code"
]

X = df_model[num_features + cat_features]
y = df_model["Primary Cuisine"]

# ------------------------------------------------------------
# 6. ENCODE CATEGORICAL VARIABLES
# ------------------------------------------------------------

# ColumnTransformer lets us apply different preprocessing
# to numeric and categorical columns at once.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# ------------------------------------------------------------
# 7. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # keep class proportions in train & test
)

print(f"\nTraining rows: {X_train.shape[0]}")
print(f"Testing rows : {X_test.shape[0]}")

# ------------------------------------------------------------
# 8. TRAIN MULTIPLE MODELS
# ------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, n_jobs=-1),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "Naive Bayes": MultinomialNB()
}

results = {}
trained_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", clf)
    ])

    # Naive Bayes works on non-negative counts; skip StandardScaler issue
    # by leaving it — it still usually works, or you can use a separate pipeline.
    try:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "y_pred": y_pred
        }
        trained_pipelines[name] = pipe
        print(f"\n{name} trained successfully.")
    except Exception as e:
        print(f"\n{name} failed: {e}")

# ------------------------------------------------------------
# 9. COMPARE MODELS
# ------------------------------------------------------------

comparison = pd.DataFrame({
    name: {
        "Accuracy": r["accuracy"],
        "Precision (weighted)": r["precision"],
        "Recall (weighted)": r["recall"],
        "F1 (weighted)": r["f1"]
    }
    for name, r in results.items()
}).T

print("\n===== Model Comparison =====")
print(comparison.round(4))

# Pick best model by F1
best_model_name = comparison["F1 (weighted)"].idxmax()
best_model = trained_pipelines[best_model_name]
best_pred = results[best_model_name]["y_pred"]

print(f"\nBest model: {best_model_name}")

# ------------------------------------------------------------
# 10. DETAILED CLASSIFICATION REPORT
# ------------------------------------------------------------

print(f"\n===== Classification Report ({best_model_name}) =====")
print(classification_report(y_test, best_pred, zero_division=0))

# ------------------------------------------------------------
# 11. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test, best_pred, labels=top_cuisines)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=top_cuisines,
    yticklabels=top_cuisines
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Cuisine")
plt.ylabel("Actual Cuisine")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 12. PER-CUISINE PERFORMANCE ANALYSIS
# ------------------------------------------------------------

report_dict = classification_report(
    y_test, best_pred, output_dict=True, zero_division=0
)

per_cuisine = pd.DataFrame({
    cuisine: {
        "Precision": report_dict[cuisine]["precision"],
        "Recall": report_dict[cuisine]["recall"],
        "F1-Score": report_dict[cuisine]["f1-score"],
        "Support": report_dict[cuisine]["support"]
    }
    for cuisine in top_cuisines
    if cuisine in report_dict
}).T.sort_values("F1-Score", ascending=False)

print("\n===== Per-Cuisine Performance =====")
print(per_cuisine.round(3))

# ------------------------------------------------------------
# 13. FEATURE IMPORTANCE (Random Forest)
# ------------------------------------------------------------

if "Random Forest" in trained_pipelines:
    rf_pipe = trained_pipelines["Random Forest"]
    rf_clf = rf_pipe.named_steps["classifier"]

    # Get feature names after one-hot encoding
    ohe = rf_pipe.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat_features)
    all_feature_names = num_features + list(cat_names)

    importances = rf_clf.feature_importances_
    fi_df = (
        pd.DataFrame({"Feature": all_feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis")
    plt.title("Top 15 Feature Importances - Random Forest")
    plt.tight_layout()
    plt.show()

    print("\nTop 15 important features:")
    print(fi_df)