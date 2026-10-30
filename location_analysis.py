# ============================================================
# LOCATION-BASED ANALYSIS OF RESTAURANTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Interactive maps
try:
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False
    print("folium not installed — map will be skipped.")

try:
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
    print("plotly not installed — plotly map will be skipped.")

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ------------------------------------------------------------
# 1. LOAD & CLEAN THE DATASET
# ------------------------------------------------------------

df = pd.read_csv("Dataset.csv", encoding="utf-8-sig")
print("Original shape:", df.shape)

# Drop duplicate restaurants
df = df.drop_duplicates(subset="Restaurant ID").copy()

# Convert numeric columns
for col in ["Latitude", "Longitude", "Average Cost for two",
            "Aggregate rating", "Votes", "Price range"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing text values
df["City"] = df["City"].fillna("Unknown")
df["Locality"] = df["Locality"].fillna("Unknown")
df["Cuisines"] = df["Cuisines"].fillna("Unknown")
df["Country Code"] = df["Country Code"].fillna(0)

# Fill missing numerics
df["Average Cost for two"] = df["Average Cost for two"].fillna(
    df["Average Cost for two"].median()
)
df["Aggregate rating"] = df["Aggregate rating"].fillna(0)
df["Votes"] = df["Votes"].fillna(0)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 2. BASIC GEOGRAPHIC OVERVIEW
# ------------------------------------------------------------

# Some rows have (0,0) coordinates which are invalid.
invalid_coords = df[(df["Latitude"] == 0) | (df["Longitude"] == 0)]
print(f"\nRows with (0,0) coordinates: {len(invalid_coords)}")

# Drop them for mapping, keep the rest for stats
df_geo = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)].copy()

print(f"Rows with valid coordinates: {len(df_geo)}")
print("\nLatitude range:", df_geo["Latitude"].min(), "→", df_geo["Latitude"].max())
print("Longitude range:", df_geo["Longitude"].min(), "→", df_geo["Longitude"].max())

# Country breakdown
print("\nRestaurants per country code:")
print(df_geo["Country Code"].value_counts())

# ------------------------------------------------------------
# 3. GLOBAL SCATTER OF RESTAURANTS
# ------------------------------------------------------------

plt.figure(figsize=(14, 8))
plt.scatter(
    df_geo["Longitude"],
    df_geo["Latitude"],
    s=8,
    alpha=0.4,
    c=df_geo["Aggregate rating"],
    cmap="viridis"
)
plt.colorbar(label="Aggregate Rating")
plt.title("Global Distribution of Restaurants (colored by rating)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. INTERACTIVE MAP WITH FOLIUM
# ------------------------------------------------------------

if FOLIUM_OK:
    # Center the map on the mean coordinates
    center_lat = df_geo["Latitude"].mean()
    center_lon = df_geo["Longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=2,
                   tiles="CartoDB positron")

    # Cluster markers so thousands of points don't slow the browser
    marker_cluster = MarkerCluster(name="Restaurants").add_to(m)

    # Sample up to 2000 restaurants so the HTML isn't huge
    sample = df_geo.sample(min(2000, len(df_geo)), random_state=42)

    for _, row in sample.iterrows():
        popup = (
            f"<b>{row['Restaurant Name']}</b><br>"
            f"City: {row['City']}<br>"
            f"Cuisine: {row['Cuisines']}<br>"
            f"Rating: {row['Aggregate rating']}<br>"
            f"Cost for two: {row['Average Cost for two']} {row['Currency']}"
        )
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=3,
            color="crimson",
            fill=True,
            fill_opacity=0.7,
            popup=popup
        ).add_to(marker_cluster)

    # Heatmap layer showing density
    heat_data = df_geo[["Latitude", "Longitude"]].values.tolist()
    HeatMap(heat_data, radius=6, blur=8, name="Density").add_to(m)

    folium.LayerControl().add_to(m)
    m.save("restaurants_map.html")
    print("\nInteractive map saved as restaurants_map.html — open it in a browser.")

# ------------------------------------------------------------
# 5. PLOTLY MAP (fallback / extra)
# ------------------------------------------------------------

if PLOTLY_OK:
    sample_plot = df_geo.sample(min(3000, len(df_geo)), random_state=1)

    # Try new API first (Plotly 6+), fall back to old API if needed
    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(
            sample_plot,
            lat="Latitude",
            lon="Longitude",
            color="Aggregate rating",
            size="Votes",
            hover_name="Restaurant Name",
            hover_data=["City", "Cuisines", "Average Cost for two"],
            color_continuous_scale="Viridis",
            zoom=1,
            height=600,
            title="Interactive Restaurant Map (Plotly)"
        )
        fig.update_layout(map_style="carto-positron")
    else:
        # Old Plotly (< 6.0)
        fig = px.scatter_mapbox(
            sample_plot,
            lat="Latitude",
            lon="Longitude",
            color="Aggregate rating",
            size="Votes",
            hover_name="Restaurant Name",
            hover_data=["City", "Cuisines", "Average Cost for two"],
            color_continuous_scale="Viridis",
            zoom=1,
            height=600,
            title="Interactive Restaurant Map (Plotly)"
        )
        fig.update_layout(mapbox_style="carto-positron")

    fig.show()

# ------------------------------------------------------------
# 6. RESTAURANT CONCENTRATION BY CITY
# ------------------------------------------------------------

city_counts = (
    df_geo.groupby("City")
    .size()
    .sort_values(ascending=False)
)

print("\nTop 20 cities by restaurant count:")
print(city_counts.head(20))

plt.figure(figsize=(12, 7))
city_counts.head(15).sort_values().plot(kind="barh", color="steelblue")
plt.title("Top 15 Cities by Number of Restaurants")
plt.xlabel("Number of Restaurants")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. RESTAURANT CONCENTRATION BY LOCALITY (WITHIN CITY)
# ------------------------------------------------------------

locality_counts = (
    df_geo.groupby(["City", "Locality"])
    .size()
    .sort_values(ascending=False)
    .reset_index(name="count")
)

print("\nTop 20 localities overall:")
print(locality_counts.head(20))

# Example: localities inside a specific city
sample_city = "Makati City"
if sample_city in df_geo["City"].unique():
    print(f"\nLocalities in {sample_city}:")
    print(
        df_geo[df_geo["City"] == sample_city]
        .groupby("Locality")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

# ------------------------------------------------------------
# 8. CITY-LEVEL STATISTICS
# ------------------------------------------------------------

def top_cuisine(series):
    """Return the most common cuisine in a group."""
    return (
        series.dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .idxmax()
        if len(series.dropna()) > 0 else "Unknown"
    )

city_stats = df_geo.groupby("City").agg(
    Total_Restaurants=("Restaurant ID", "count"),
    Avg_Rating=("Aggregate rating", "mean"),
    Avg_Cost=("Average Cost for two", "mean"),
    Median_Cost=("Average Cost for two", "median"),
    Avg_Votes=("Votes", "mean"),
    Avg_Price_Range=("Price range", "mean"),
).round(2)

# Add most common cuisine per city
city_stats["Top_Cuisine"] = df_geo.groupby("City")["Cuisines"].apply(top_cuisine)
city_stats["Top_Cuisine"] = city_stats["Top_Cuisine"].fillna("Unknown")

# Only show cities with at least 10 restaurants
city_stats = city_stats[city_stats["Total_Restaurants"] >= 10]
city_stats = city_stats.sort_values("Total_Restaurants", ascending=False)

print("\n===== City-level statistics =====")
print(city_stats.head(20))

# ------------------------------------------------------------
# 9. VISUALISE CITY-LEVEL AVERAGES
# ------------------------------------------------------------

top_cities = city_stats.head(12)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Average rating
sns.barplot(
    x=top_cities["Avg_Rating"],
    y=top_cities.index,
    ax=axes[0],
    palette="Greens_r"
)
axes[0].set_title("Average Rating by City")
axes[0].set_xlabel("Avg Rating")

# Average cost
sns.barplot(
    x=top_cities["Avg_Cost"],
    y=top_cities.index,
    ax=axes[1],
    palette="Oranges_r"
)
axes[1].set_title("Average Cost for Two by City")
axes[1].set_xlabel("Avg Cost for Two")

# Restaurant count
sns.barplot(
    x=top_cities["Total_Restaurants"],
    y=top_cities.index,
    ax=axes[2],
    palette="Blues_r"
)
axes[2].set_title("Number of Restaurants by City")
axes[2].set_xlabel("Count")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10. PRICE RANGE DISTRIBUTION BY CITY
# ------------------------------------------------------------

# Price range distribution for top cities
top_city_list = city_stats.head(8).index.tolist()
subset = df_geo[df_geo["City"].isin(top_city_list)]

plt.figure(figsize=(12, 6))
sns.countplot(
    data=subset,
    x="City",
    hue="Price range",
    order=top_city_list,
    palette="Set2"
)
plt.title("Price Range Distribution Across Top Cities")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 11. TOP CUISINES IN THE TOP CITIES
# ------------------------------------------------------------

# Melt cuisines so we can count individual cuisines per city
cuisine_exploded = (
    df_geo.assign(Cuisine=df_geo["Cuisines"].str.split(","))
    .explode("Cuisine")
)
cuisine_exploded["Cuisine"] = cuisine_exploded["Cuisine"].str.strip()

top_cuisines_per_city = (
    cuisine_exploded[cuisine_exploded["City"].isin(top_city_list)]
    .groupby(["City", "Cuisine"])
    .size()
    .reset_index(name="Count")
)

# Pick the top 5 cuisines per city
top5 = (
    top_cuisines_per_city
    .sort_values(["City", "Count"], ascending=[True, False])
    .groupby("City")
    .head(5)
)

print("\n===== Top 5 Cuisines per City =====")
for city in top_city_list:
    print(f"\n{city}:")
    print(top5[top5["City"] == city][["Cuisine", "Count"]].to_string(index=False))

# ------------------------------------------------------------
# 12. RATING DENSITY BY LATITUDE / LONGITUDE (HEXBIN)
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))
hb = plt.hexbin(
    df_geo["Longitude"],
    df_geo["Latitude"],
    C=df_geo["Aggregate rating"],
    gridsize=60,
    cmap="coolwarm",
    reduce_C_function=np.mean,
    mincnt=1
)
plt.colorbar(hb, label="Average Rating")
plt.title("Heatmap of Average Rating by Geographic Area")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 13. CORRELATION: COST VS RATING PER CITY
# ------------------------------------------------------------

top3_cities = city_stats.head(3).index.tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, city in zip(axes, top3_cities):
    sub = df_geo[df_geo["City"] == city]
    ax.scatter(sub["Average Cost for two"], sub["Aggregate rating"],
               alpha=0.4, s=15, color="teal")
    ax.set_title(f"{city}\nCost vs Rating")
    ax.set_xlabel("Average Cost for two")
    ax.set_ylabel("Rating")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 14. INSIGHTS SUMMARY (printed, not hard-coded)
# ------------------------------------------------------------

print("\n================ LOCATION INSIGHTS ================")

most_restaurants = city_counts.idxmax()
print(f"• City with the most restaurants: {most_restaurants} "
      f"({city_counts.max()} restaurants)")

best_rated = city_stats["Avg_Rating"].idxmax()
print(f"• Highest average rated city: {best_rated} "
      f"({city_stats.loc[best_rated, 'Avg_Rating']:.2f})")

most_expensive = city_stats["Avg_Cost"].idxmax()
print(f"• Most expensive city (avg cost for two): {most_expensive} "
      f"({city_stats.loc[most_expensive, 'Avg_Cost']:.0f})")

cheapest = city_stats["Avg_Cost"].idxmin()
print(f"• Cheapest city (avg cost for two): {cheapest} "
      f"({city_stats.loc[cheapest, 'Avg_Cost']:.0f})")

print("\n• Top 5 cities and their most common cuisine:")
for city in city_stats.head(5).index:
    print(f"   - {city} → {city_stats.loc[city, 'Top_Cuisine']}")