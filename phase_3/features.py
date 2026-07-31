import numpy as np
import pandas as pd
import sys
sys.path.append("../phase_1")
sys.path.append("../phase_2")
from data import load_ratings, load_movies
from content_based import get_genres

GENRES = None  # set on first call to build_movie_vectors

def get_genre_matrix(movies):
    """
    Returns (movie_ids array, genre_matrix of shape [n_movies, n_genres])
    Fully vectorized — no iterrows.
    """
    global GENRES
    genres = get_genres(movies)
    GENRES = genres

    # one-hot encode all movies at once
    rows = []
    for _, row in movies.iterrows():
        rows.append([1.0 if g in row["genre_list"] else 0.0 for g in genres])

    movie_ids = movies["movie_id"].values
    genre_matrix = np.array(rows, dtype=np.float32)  # (n_movies, 18)
    return movie_ids, genre_matrix

def build_user_features(ratings, movies):
    """
    Vectorized user features:
      - avg_rating    : normalised mean rating given
      - activity      : log-normalised number of ratings
      - genre_profile : weighted average of genre vectors for rated movies
    """
    movie_ids, genre_matrix = get_genre_matrix(movies)
    mid_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    # map movie_id → genre vector index
    valid = ratings["movie_id"].isin(mid_to_idx)
    r = ratings[valid].copy()
    r["item_idx"] = r["movie_id"].map(mid_to_idx)

    user_features = {}
    grouped = r.groupby("user_id")

    for user_id, group in grouped:
        weights = group["rating"].values.astype(np.float32)      # (n,)
        idxs    = group["item_idx"].values                        # (n,)
        vecs    = genre_matrix[idxs]                              # (n, 18)

        profile     = (weights[:, None] * vecs).sum(axis=0) / weights.sum()
        avg_rating  = weights.mean() / 5.0
        activity    = np.log1p(len(group)) / 10.0

        user_features[user_id] = np.concatenate([
            [avg_rating, activity],
            profile,
        ]).astype(np.float32)  # (20,)

    return user_features

def build_item_features(movies, ratings):
    """
    Vectorized item features:
      - year_norm   : release year normalised to [0,1]
      - pop_norm    : log-normalised popularity
      - genre vec   : 18-dim binary
    """
    movie_ids, genre_matrix = get_genre_matrix(movies)
    popularity = ratings.groupby("movie_id").size()

    def extract_year(title):
        try:
            return int(title.strip()[-5:-1])
        except:
            return 1990

    years = movies["title"].apply(extract_year).values
    year_norm = ((years - 1920) / (2000 - 1920)).clip(0, 1).astype(np.float32)
    pop_norm  = np.log1p(
        movies["movie_id"].map(popularity).fillna(0).values
    ).astype(np.float32) / 10.0

    item_features = {}
    for i, mid in enumerate(movie_ids):
        item_features[mid] = np.concatenate([
            [year_norm[i], pop_norm[i]],
            genre_matrix[i],
        ])  # (20,)

    return item_features

def build_training_pair(user_id, movie_id, user_features, item_features):
    u = user_features.get(user_id, np.zeros(20, dtype=np.float32))
    i = item_features.get(movie_id, np.zeros(20, dtype=np.float32))
    return np.concatenate([u, i])  # (40,)

if __name__ == "__main__":
    ratings = load_ratings()
    movies  = load_movies()

    print("Building user features...")
    user_features = build_user_features(ratings, movies)
    print(f"  Done — {len(user_features)} users")

    print("Building item features...")
    item_features = build_item_features(movies, ratings)
    print(f"  Done — {len(item_features)} movies")

    uid = list(user_features.keys())[0]
    mid = list(item_features.keys())[0]
    print(f"\nUser {uid} vector: {user_features[uid].round(3)}")
    print(f"Movie {mid} vector: {item_features[mid].round(3)}")
    print(f"Combined pair: {build_training_pair(uid, mid, user_features, item_features).shape} dims")
