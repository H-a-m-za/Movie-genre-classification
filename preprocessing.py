import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import resample

# Load data
df = pd.read_csv('TMDB_all_movies.csv')
df = df[['title', 'overview', 'tagline', 'genres']]
df = df.dropna()

# Combine text fields
df['combined_text'] = df['title'] + " " + df['overview'].fillna('') + " " + df['tagline'].fillna('')
df['combined_text'] = df['combined_text'].astype(str)

# Parse and clean genres
def parse_genres(genres_str):
    if isinstance(genres_str, str):
        try:
            return [genre.strip() for genre in genres_str.split(',')]
        except Exception as e:
            print(f"Error parsing genres: {e}")
            return []
    return []

df['genres'] = df['genres'].apply(parse_genres)

# Binarize genres
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])

# Create a DataFrame to store the genres and corresponding combined text
df_binarized = pd.DataFrame(y, columns=mlb.classes_)
df_binarized['combined_text'] = df['combined_text']

# Balance the dataset by resampling
balanced_dfs = []

min_samples = df_binarized.drop(columns=['combined_text']).sum().min()

for genre in mlb.classes_:
    # Select samples where the genre is present
    genre_df = df_binarized[df_binarized[genre] == 1]

    # Resample to match the least represented genre
    balanced_df = resample(genre_df, replace=False, n_samples=min_samples, random_state=42)
    balanced_dfs.append(balanced_df)

# Combine balanced dataframes
balanced_df = pd.concat(balanced_dfs)

# Shuffle the balanced dataframe
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the distribution after balancing
balanced_genre_counts = balanced_df.drop(columns=['combined_text']).sum().sort_values(ascending=False)
print(balanced_genre_counts)

# Print the number of unique genres
print(f"Number of unique genres after balancing: {len(balanced_genre_counts)}")

# Plot the balanced genre distribution
plt.figure(figsize=(12, 6))
balanced_genre_counts.plot(kind='bar')
plt.title('Balanced Genre Distribution')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()
# Save the balanced dataframe to a CSV file
df.to_csv('TMDB_balanced_movies.csv', index=False)
