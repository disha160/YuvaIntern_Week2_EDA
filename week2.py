import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Load the Netflix dataset
df = pd.read_csv("netflix_titles.csv")

# Display the first 5 rows
print("FIRST 5 ROWS")
print(df.head())

# Display dataset shape
print("\nDATASET SHAPE")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Display data types
print("\nDATA TYPES")
print(df.dtypes)

# Display basic information
print("\nDATASET INFORMATION")
df.info()

# Display statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())
# --------------------------------------------------
# 3. MISSING VALUE ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

# Count missing values
missing_values = df.isnull().sum()

# Calculate missing percentage
missing_percentage = (missing_values / len(df)) * 100

# Create summary table
missing_summary = pd.DataFrame({
    "Missing Values": missing_values,
    "Percentage": missing_percentage
})

# Display only columns containing missing values
missing_summary = missing_summary[
    missing_summary["Missing Values"] > 0
]

print("\nMissing Value Summary:")
print(missing_summary)


# --------------------------------------------------
# 4. DUPLICATE ROW ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE ROW ANALYSIS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate rows:", duplicate_count)
# --------------------------------------------------
# 5. CATEGORICAL DATA ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 60)
print("CATEGORICAL DATA ANALYSIS")
print("=" * 60)

# 1. Movies vs TV Shows
print("\n1. CONTENT TYPE")
print(df["type"].value_counts())

# 2. Content ratings
print("\n2. CONTENT RATINGS")
print(df["rating"].value_counts(dropna=False))

# 3. Top 10 countries
print("\n3. TOP 10 COUNTRIES")
print(df["country"].value_counts().head(10))

# 4. Top 10 genres
print("\n4. TOP 10 GENRES")

# Each title can have multiple genres separated by commas
genres = df["listed_in"].dropna().str.split(", ").explode()

print(genres.value_counts().head(10))
# --------------------------------------------------
# 6. NUMERICAL AND TIME-BASED ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL AND TIME-BASED ANALYSIS")
print("=" * 60)

# 1. Release year analysis
print("\n1. RELEASE YEAR ANALYSIS")

print("Earliest release year:", df["release_year"].min())
print("Latest release year:", df["release_year"].max())
print("Median release year:", df["release_year"].median())

print("\nTop 10 release years:")
print(df["release_year"].value_counts().sort_index(ascending=False).head(10))


# --------------------------------------------------
# 2. CONVERT DATE_ADDED TO DATETIME
# --------------------------------------------------

print("\n2. DATE ADDED ANALYSIS")

df["date_added"] = pd.to_datetime(
    df["date_added"],
    errors="coerce"
)

print("Earliest date added:", df["date_added"].min())
print("Latest date added:", df["date_added"].max())


# Extract year from date_added
df["year_added"] = df["date_added"].dt.year

print("\nContent added by year:")
print(df["year_added"].value_counts().sort_index())


# --------------------------------------------------
# 3. DURATION ANALYSIS
# --------------------------------------------------

print("\n3. DURATION ANALYSIS")

print("\nMost common duration values:")
print(df["duration"].value_counts().head(15))


# Separate Movies and TV Shows
movies = df[df["type"] == "Movie"].copy()
tv_shows = df[df["type"] == "TV Show"].copy()

print("\nNumber of movies:", len(movies))
print("Number of TV shows:", len(tv_shows))


# --------------------------------------------------
# 4. MOVIE DURATION IN MINUTES
# --------------------------------------------------

print("\n4. MOVIE DURATION ANALYSIS")

movies["duration_minutes"] = (
    movies["duration"]
    .str.replace(" min", "", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
)

print("\nMovie duration statistics:")
print(movies["duration_minutes"].describe())


# --------------------------------------------------
# 5. TV SHOW SEASONS
# --------------------------------------------------

print("\n5. TV SHOW SEASON ANALYSIS")

tv_shows["seasons"] = (
    tv_shows["duration"]
    .str.extract(r"(\d+)")
    .astype(float)
)

print("\nTV Show season statistics:")
print(tv_shows["seasons"].describe())

print("\nMost common number of seasons:")
print(tv_shows["seasons"].value_counts().sort_index())

# --------------------------------------------------
# 7. ANOMALY INVESTIGATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("ANOMALY INVESTIGATION")
print("=" * 60)

# Find unusual rating values
unusual_ratings = ["66 min", "74 min", "84 min"]

print("\nUnusual rating entries:")
print(df[df["rating"].isin(unusual_ratings)][
    ["title", "type", "rating", "duration"]
])


# --------------------------------------------------
# CHECK UNUSUAL MOVIE DURATIONS
# --------------------------------------------------

print("\nUnusually short movies (less than 10 minutes):")

short_movies = movies[movies["duration_minutes"] < 10]

print(short_movies[
    ["title", "duration", "rating"]
])
# --------------------------------------------------
# 8. VISUALIZATION 1 - MOVIES VS TV SHOWS
# --------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="type")

plt.title("Distribution of Movies and TV Shows")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.show()
# --------------------------------------------------
# 9. VISUALIZATION 2 - TOP 10 CONTENT RATINGS
# --------------------------------------------------

plt.figure(figsize=(10, 6))

top_ratings = df["rating"].value_counts().head(10)

sns.barplot(x=top_ratings.values, y=top_ratings.index)

plt.title("Top 10 Content Ratings")
plt.xlabel("Number of Titles")
plt.ylabel("Rating")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 10. VISUALIZATION 3 - TOP 10 COUNTRIES
# --------------------------------------------------

plt.figure(figsize=(10, 6))

top_countries = df["country"].value_counts().head(10)

sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)

plt.title("Top 10 Countries by Number of Titles")
plt.xlabel("Number of Titles")
plt.ylabel("Country")

plt.tight_layout()
plt.show()
# --------------------------------------------------
# 11. VISUALIZATION 4 - TOP 10 GENRES
# --------------------------------------------------

genres = df["listed_in"].dropna().str.split(", ")

genre_counts = genres.explode().value_counts().head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=genre_counts.values,
    y=genre_counts.index
)

plt.title("Top 10 Genres")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")

plt.tight_layout()
plt.show()
# --------------------------------------------------
# 12. VISUALIZATION 5 - CONTENT ADDED BY YEAR
# --------------------------------------------------

date_added = pd.to_datetime(df["date_added"], errors="coerce")

year_added = date_added.dt.year.value_counts().sort_index()

plt.figure(figsize=(10, 6))

sns.lineplot(
    x=year_added.index,
    y=year_added.values,
    marker="o"
)

plt.title("Number of Titles Added by Year")
plt.xlabel("Year")
plt.ylabel("Number of Titles Added")

plt.tight_layout()
plt.show()
# --------------------------------------------------
# 13. VISUALIZATION 6 - MOVIE DURATION DISTRIBUTION
# --------------------------------------------------

movie_duration = df[df["type"] == "Movie"]["duration"].str.extract(
    r"(\d+)"
)[0].astype(float)

plt.figure(figsize=(10, 6))

sns.histplot(
    movie_duration.dropna(),
    bins=30,
    kde=True
)

plt.title("Distribution of Movie Durations")
plt.xlabel("Duration (Minutes)")
plt.ylabel("Number of Movies")

plt.tight_layout()
plt.show()
# --------------------------------------------------
# 14. VISUALIZATION 7 - TV SHOW SEASON DISTRIBUTION
# --------------------------------------------------

season_counts = tv_shows["seasons"].value_counts().sort_index()

plt.figure(figsize=(10, 6))

sns.barplot(
    x=season_counts.index,
    y=season_counts.values
)

plt.title("Distribution of TV Show Seasons")
plt.xlabel("Number of Seasons")
plt.ylabel("Number of TV Shows")

plt.tight_layout()
plt.show()