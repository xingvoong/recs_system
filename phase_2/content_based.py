import numpy as np
import pandas as pd
import sys
sys.path.append("../phase_1")
from data import load_movies, load_ratings

def get_genres(movies):
    return sorted(movies["genre_list"].explode().dropna().unique())

def build_movie_vectors(movies):
    """
    Each movie → binary genre vector derived from the data.
    Toy Story (Animation|Children's|Comedy) → [0,0,1,1,1,0,0,0,...]
    """
    genres = get_genres(movies)
    vectors = {}
    for _, row in movies.iterrows():
        vec = np.array([1.0 if g in row["genre_list"] else 0.0 for g in genres])
        vectors[row["movie_id"]] = vec
    return vectors  # {movie_id: np.array(n_genres,)}

def build_user_profile(user_id, ratings, movie_vectors):
    """
    User profile = weighted average of genre vectors for movies they rated.
    Higher-rated movies contribute more to the profile.
    """
    user_ratings = ratings[ratings["user_id"] == user_id]
    if user_ratings.empty:
        return None

    n_dims = len(next(iter(movie_vectors.values())))
    profile = np.zeros(n_dims)
    total_weight = 0.0

    for _, row in user_ratings.iterrows():
        mid = row["movie_id"]
        if mid not in movie_vectors:
            continue
        weight = row["rating"]
        profile += weight * movie_vectors[mid]
        total_weight += weight

    if total_weight == 0:
        return None

    return profile / total_weight  # normalise

def generate_candidates(user_id, ratings, movies, movie_vectors, top_n=100):
    """
    Score every movie by dot product with the user profile.
    Return top_n movie_ids the user hasn't already rated.
    """
    profile = build_user_profile(user_id, ratings, movie_vectors)
    if profile is None:
        return []

    seen = set(ratings[ratings["user_id"] == user_id]["movie_id"])

    scores = {}
    for mid, vec in movie_vectors.items():
        if mid in seen:
            continue
        scores[mid] = float(np.dot(profile, vec))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in ranked[:top_n]]

if __name__ == "__main__":
    movies  = load_movies()
    ratings = load_ratings()
    genres  = get_genres(movies)
    movie_vectors = build_movie_vectors(movies)

    print(f"Genres derived from data: {genres}")
    print(f"Movie vectors built: {len(movie_vectors)} movies, {len(genres)} dims")

    # inspect a few vectors
    print("\nSample movie vectors:")
    for mid in [1, 6, 1617]:   # Toy Story, Heat, L.A. Confidential
        title = movies.set_index("movie_id").loc[mid, "title"]
        vec   = movie_vectors[mid]
        active = [genres[i] for i, v in enumerate(vec) if v]
        print(f"  [{mid}] {title}")
        print(f"       genres: {active}")

    # generate candidates for a sample user
    user_id = 1
    candidates = generate_candidates(user_id, ratings, movies, movie_vectors, top_n=10)
    movie_info = movies.set_index("movie_id")

    print(f"\nTop 10 content-based candidates for user {user_id}:")
    profile = build_user_profile(user_id, ratings, movie_vectors)
    print(f"  User profile (top genres): {[genres[i] for i in np.argsort(profile)[::-1][:5]]}")
    print()
    for mid in candidates:
        title  = movie_info.loc[mid, "title"] if mid in movie_info.index else "?"
        genres_str = movie_info.loc[mid, "genres"] if mid in movie_info.index else "?"
        print(f"  {title}  ({genres_str})")
