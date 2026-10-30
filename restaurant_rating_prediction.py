# ============================================================
# PREDICT RESTAURANT RATINGS - REGRESSION MODEL
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("Dataset.csv", encoding="utf-8-sig")

print("Original shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 2. CLEANING
# ------------------------------------------------------------

# Remove duplicates by Restaurant ID
df = df.drop_duplicates(subset="Restaurant ID").copy()

# Convert numeric columns
numeric_cols = [
    "Average Cost for two", "Aggregate rating",
    "Votes", "Price range", "Latitude", "Longitude"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill text columns
df["Cuisines"] = df["Cuisines"].fillna("Unknown")
df["City"] = df["City"].fillna("Unknown")
df["Locality"] = df["Locality"].fillna("Unknown")
df["Has Table booking"] = df["Has Table booking"].fillna("No")
df["Has Online delivery"] = df["Has Online delivery"].fillna("No")
df["Country Code"] = df["Country Code"].fillna(0).astype(int)

# ------------------------------------------------------------
# 3. TARGET VARIABLE
# ------------------------------------------------------------

# Aggregate rating = 0 means "Not rated" — these rows can't help us learn.
# Drop them from training.
before = len(df)
df = df[df["Aggregate rating"] > 0].copy()
after = len(df)
print(f"\nDropped {before - after} rows with rating = 0 (Not rated).")
print(f"Remaining rows: {after}")

print("\nRating distribution:")
print(df["Aggregate rating"].describe())

# Visualise the target distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["Aggregate rating"], bins=25, kde=True, color="teal")
plt.title("Distribution of Aggregate Ratings")
plt.xlabel("Aggregate Rating")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ------------------------------------------------------------

# Number of cuisines listed (multi-cuisine restaurants may behave differently)
df["Cuisine Count"] = df["Cuisines"].apply(
    lambda x: len(str(x).split(","))
)

# Primary cuisine (first listed) — reduces cardinality vs full list
df["Primary Cuisine"] = df["Cuisines"].apply(
    lambda x: str(x).split(",")[0].strip()
)

# Service flags as binary
df["Has Table booking"] = df["Has Table booking"].map(
    {"Yes": 1, "No": 0}
).fillna(0).astype(int)

df["Has Online delivery"] = df["Has Online delivery"].map(
    {"Yes": 1, "No": 0}
).fillna(0).astype(int)

# ------------------------------------------------------------
# 5. SELECT FEATURES AND TARGET
# ------------------------------------------------------------

numeric_features = [
    "Average Cost for two",
    "Votes",
    "Price range",
    "Cuisine Count",
    "Latitude",
    "Longitude",
    "Has Table booking",
    "Has Online delivery"
]

categorical_features = [
    "City",
    "Primary Cuisine",
    "Country Code"
]

X = df[numeric_features + categorical_features]
y = df["Aggregate rating"]

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

# ------------------------------------------------------------
# 6. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining rows: {X_train.shape[0]}")
print(f"Testing rows : {X_test.shape[0]}")

# ------------------------------------------------------------
# 7. PREPROCESSING PIPELINE
# ------------------------------------------------------------

# Numeric: median imputation + standard scaling
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical: most frequent imputation + one-hot encoding
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ------------------------------------------------------------
# 8. DEFINE MODELS
# ------------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, random_state=42
    )
}

# ------------------------------------------------------------
# 9. TRAIN AND EVALUATE
# ------------------------------------------------------------

results = {}
trained_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "y_pred": y_pred
    }
    trained_pipelines[name] = pipe

    print(f"\n--- {name} ---")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

# ------------------------------------------------------------
# 10. COMPARE MODELS
# ------------------------------------------------------------

comparison = pd.DataFrame({
    name: {
        "MSE": r["MSE"],
        "RMSE": r["RMSE"],
        "MAE": r["MAE"],
        "R²": r["R2"]
    }
    for name, r in results.items()
}).T.sort_values("R²", ascending=False)

print("\n===== Model Comparison =====")
print(comparison.round(4))

# Bar chart of R²
plt.figure(figsize=(10, 5))
sns.barplot(x=comparison.index, y=comparison["R²"], palette="viridis")
plt.title("Model Comparison by R²")
plt.ylabel("R² Score")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# Pick best model
best_model_name = comparison["R²"].idxmax()
best_model = trained_pipelines[best_model_name]
best_pred = results[best_model_name]["y_pred"]

print(f"\nBest model: {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")

# ------------------------------------------------------------
# 11. ACTUAL VS PREDICTED
# ------------------------------------------------------------

plt.figure(figsize=(8, 8))
plt.scatter(y_test, best_pred, alpha=0.3, s=15, color="teal")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--", linewidth=2
)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title(f"Actual vs Predicted ({best_model_name})")
plt.tight_layout()
plt.show()

# Residual plot
residuals = y_test - best_pred

plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=40, kde=True, color="coral")
plt.title("Residual Distribution")
plt.xlabel("Actual − Predicted")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 12. FEATURE IMPORTANCE
# ------------------------------------------------------------

# Use Random Forest because it gives direct feature importances
rf_pipe = trained_pipelines["Random Forest"]
rf_model = rf_pipe.named_steps["regressor"]

# Get feature names after one-hot encoding
ohe = rf_pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
cat_names = ohe.get_feature_names_out(categorical_features)
all_names = numeric_features + list(cat_names)

importances = rf_model.feature_importances_
fi_df = (
    pd.DataFrame({"Feature": all_names, "Importance": importances})
    .sort_values("Importance", ascending=False)
    .head(20)
)

plt.figure(figsize=(10, 8))
sns.barplot(data=fi_df, x="Importance", y="Feature", palette="mako")
plt.title("Top 20 Most Influential Features (Random Forest)")
plt.tight_layout()
plt.show()

print("\nTop 20 influential features:")
print(fi_df.to_string(index=False))

# ------------------------------------------------------------
# 13. SIMPLE INTERPRETATION VIA CORRELATION
# ------------------------------------------------------------

# Correlation of numeric features with target
corr_df = df[numeric_features + ["Aggregate rating"]].corr()
rating_corr = corr_df["Aggregate rating"].drop("Aggregate rating").sort_values()

plt.figure(figsize=(8, 6))
rating_corr.plot(kind="barh", color="steelblue")
plt.title("Correlation of Numeric Features with Aggregate Rating")
plt.xlabel("Correlation")
plt.tight_layout()
plt.show()

print("\nCorrelation with Aggregate Rating:")
print(rating_corr.round(3))