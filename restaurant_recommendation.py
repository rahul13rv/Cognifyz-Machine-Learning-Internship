# ============================================================
# RESTAURANT RECOMMENDATION SYSTEM - CONTENT BASED FILTERING
# ============================================================

import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

# encoding='utf-8-sig' handles the BOM character at the start of the CSV.
# If you get a UnicodeDecodeError, try encoding='latin-1'.
df = pd.read_csv("Dataset.csv", encoding="utf-8-sig")

print("Original shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 2. BASIC CLEANING
# ------------------------------------------------------------

# Remove duplicate restaurants by Restaurant ID
df = df.drop_duplicates(subset="Restaurant ID").copy()

# Convert numeric columns safely
numeric_cols = [
    "Average Cost for two",
    "Aggregate rating",
    "Votes",
    "Price range",
    "Longitude",
    "Latitude"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle missing values
df["Cuisines"] = df["Cuisines"].fillna("Unknown")
df["City"] = df["City"].fillna("Unknown")
df["Locality"] = df["Locality"].fillna("Unknown")
df["Has Table booking"] = df["Has Table booking"].fillna("No")
df["Has Online delivery"] = df["Has Online delivery"].fillna("No")

# Fill numeric missing values with median / zero
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
# 3. TEXT CLEANING FUNCTION
# ------------------------------------------------------------

def clean_text(text):
    """
    Lowercase, remove unwanted symbols, and squeeze extra spaces.
    Keeps letters, numbers, spaces, commas and ampersands.
    """
    text = str(text).lower()
    text = re.sub(r"[^\w\s,&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Clean important text columns
df["Cuisines_clean"] = df["Cuisines"].apply(clean_text).str.replace(",", " ")
df["City_clean"] = df["City"].apply(clean_text)
df["Locality_clean"] = df["Locality"].apply(clean_text)

# ------------------------------------------------------------
# 4. FEATURE ENGINEERING / CATEGORICAL ENCODING
# ------------------------------------------------------------

# Price bucket
def price_bucket(cost):
    if pd.isna(cost):
        return "unknown"
    if cost <= 500:
        return "budget"
    elif cost <= 1000:
        return "mid"
    elif cost <= 2000:
        return "premium"
    else:
        return "luxury"

df["price_bucket"] = df["Average Cost for two"].apply(price_bucket)

# Rating bucket
def rating_bucket(rating):
    if pd.isna(rating):
        return "unrated"
    if rating >= 4.5:
        return "excellent"
    elif rating >= 4.0:
        return "very_good"
    elif rating >= 3.0:
        return "good"
    else:
        return "average"

df["rating_bucket"] = df["Aggregate rating"].apply(rating_bucket)

# Binary tokens for table booking and online delivery
df["booking_token"] = np.where(
    df["Has Table booking"].str.lower() == "yes",
    "table_booking",
    ""
)

df["delivery_token"] = np.where(
    df["Has Online delivery"].str.lower() == "yes",
    "online_delivery",
    ""
)

# ------------------------------------------------------------
# 5. BUILD "SOUP" - COMBINED FEATURES FOR CONTENT FILTERING
# ------------------------------------------------------------

df["soup"] = (
    df["Cuisines_clean"] + " " +
    df["City_clean"] + " " +
    df["Locality_clean"] + " " +
    df["price_bucket"] + " " +
    df["rating_bucket"] + " " +
    df["booking_token"] + " " +
    df["delivery_token"]
)

print("\nExample soup:")
print(df[["Restaurant Name", "soup"]].head(3))

# ------------------------------------------------------------
# 6. TF-IDF VECTORIZATION
# ------------------------------------------------------------

# TF-IDF converts the soup text into numerical vectors.
# ngram_range=(1,2) captures single words and two-word phrases.
tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1
)

tfidf_matrix = tfidf.fit_transform(df["soup"])

print("\nTF-IDF matrix shape:", tfidf_matrix.shape)

# ------------------------------------------------------------
# 7. RECOMMENDATION FUNCTION
# ------------------------------------------------------------

def recommend_restaurants(user_prefs, top_n=5):
    """
    user_prefs example:
    {
        "cuisines": ["Japanese", "Sushi"],
        "city": "Makati City",          # use None if no city preference
        "max_cost": 1500,
        "min_rating": 4.0,
        "min_votes": 50,
        "table_booking": False,
        "online_delivery": False
    }
    """

    cuisines = user_prefs.get("cuisines", [])
    city = user_prefs.get("city", None)
    max_cost = user_prefs.get("max_cost", 10**9)
    min_rating = user_prefs.get("min_rating", 0)
    min_votes = user_prefs.get("min_votes", 0)
    need_booking = user_prefs.get("table_booking", False)
    need_delivery = user_prefs.get("online_delivery", False)

    # Build user soup
    cuisine_text = " ".join([clean_text(c) for c in cuisines])
    city_text = clean_text(city) if city else ""

    # Convert budget to price bucket
    if max_cost <= 500:
        price = "budget"
    elif max_cost <= 1000:
        price = "mid"
    elif max_cost <= 2000:
        price = "premium"
    else:
        price = "luxury"

    # Convert minimum rating to rating bucket
    if min_rating >= 4.5:
        rating = "excellent"
    elif min_rating >= 4.0:
        rating = "very_good"
    elif min_rating >= 3.0:
        rating = "good"
    else:
        rating = "average"

    booking = "table_booking" if need_booking else ""
    delivery = "online_delivery" if need_delivery else ""

    user_soup = f"{cuisine_text} {city_text} {price} {rating} {booking} {delivery}"

    # Transform user preferences into the same TF-IDF space
    user_vector = tfidf.transform([user_soup])

    # Cosine similarity between user profile and every restaurant
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Hard filters
    mask = (
        (df["Average Cost for two"] <= max_cost) &
        (df["Aggregate rating"] >= min_rating) &
        (df["Votes"] >= min_votes)
    )

    if city:
        mask &= (df["City_clean"] == clean_text(city))

    if need_booking:
        mask &= (df["Has Table booking"].str.lower() == "yes")

    if need_delivery:
        mask &= (df["Has Online delivery"].str.lower() == "yes")

    # Candidate restaurants
    candidates = df[mask].copy()
    candidates["similarity"] = similarity_scores[mask]

    # Sort by similarity, then rating, then votes
    candidates = candidates.sort_values(
        by=["similarity", "Aggregate rating", "Votes"],
        ascending=False
    )

    result_cols = [
        "Restaurant ID",
        "Restaurant Name",
        "City",
        "Cuisines",
        "Average Cost for two",
        "Aggregate rating",
        "Votes",
        "Has Table booking",
        "Has Online delivery",
        "similarity"
    ]

    return candidates[result_cols].head(top_n)

# ------------------------------------------------------------
# 8. TEST WITH SAMPLE USER PREFERENCES
# ------------------------------------------------------------

sample_user = {
    "cuisines": ["Japanese", "Sushi"],
    "city": None,              # try "Makati City" if you want city-specific results
    "max_cost": 1500,
    "min_rating": 4.0,
    "min_votes": 50,
    "table_booking": False,
    "online_delivery": False
}

recommendations = recommend_restaurants(sample_user, top_n=5)

print("\nTop 5 recommendations for sample user:")
print(recommendations)

# ------------------------------------------------------------
# 9. EVALUATION
# ------------------------------------------------------------

def evaluate_recommendations(user_prefs, top_n=5):
    """
    Simple evaluation:
    - Relevant restaurant = matches at least one preferred cuisine,
      within budget, rating >= minimum, and city if specified.
    - Precision@K = relevant recommended / K
    - Recall@K = relevant recommended / all relevant
    """
    recs = recommend_restaurants(user_prefs, top_n=top_n)

    pref_cuisines = set(c.lower() for c in user_prefs.get("cuisines", []))

    def has_preferred_cuisine(cuisine_string):
        cuisines = set(
            c.strip().lower()
            for c in str(cuisine_string).split(",")
        )
        return len(cuisines & pref_cuisines) > 0

    mask = (
        df["Cuisines"].apply(has_preferred_cuisine) &
        (df["Average Cost for two"] <= user_prefs.get("max_cost", 10**9)) &
        (df["Aggregate rating"] >= user_prefs.get("min_rating", 0)) &
        (df["Votes"] >= user_prefs.get("min_votes", 0))
    )

    if user_prefs.get("city"):
        mask &= (df["City_clean"] == clean_text(user_prefs["city"]))

    if user_prefs.get("table_booking"):
        mask &= (df["Has Table booking"].str.lower() == "yes")

    if user_prefs.get("online_delivery"):
        mask &= (df["Has Online delivery"].str.lower() == "yes")

    relevant_ids = set(df[mask]["Restaurant ID"])
    recommended_ids = set(recs["Restaurant ID"])

    hits = len(relevant_ids & recommended_ids)

    precision = hits / top_n if top_n > 0 else 0
    recall = hits / len(relevant_ids) if len(relevant_ids) > 0 else 0

    print("\n--- Evaluation ---")
    print(f"Total relevant restaurants: {len(relevant_ids)}")
    print(f"Recommended restaurants: {len(recommended_ids)}")
    print(f"Hits: {hits}")
    print(f"Precision@{top_n}: {precision:.2f}")
    print(f"Recall@{top_n}: {recall:.2f}")
    print(f"Average rating in top {top_n}: {recs['Aggregate rating'].mean():.2f}")
    print(f"Average cost in top {top_n}: {recs['Average Cost for two'].mean():.0f}")

    return recs

# Run evaluation
evaluated_recs = evaluate_recommendations(sample_user, top_n=5)
print("\nEvaluated recommendations:")
print(evaluated_recs)