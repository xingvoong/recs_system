import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.append("../phase_1")
from data import load_ratings, load_movies
from matrix import build_feedback_matrix

# At production scale (millions of items) you'd swap NearestNeighbors
# for FAISS (faiss.IndexFlatIP) or ScaNN. The interface is identical:
# build an index, query with a vector, get back indices + distances.
# At 3,706 movies sklearn brute-force runs in <1ms.

def build_index(item_embeddings):
    """
    Index item embeddings for fast nearest-neighbor lookup.
    metric='cosine' so we find movies with the most similar audience patterns.
    """
    index = NearestNeighbors(metric="cosine", algorithm="brute")
    index.fit(item_embeddings)
    return index

def generate_candidates(user_id, user_to_idx, idx_to_movie, ratings,
                        user_embeddings, item_embeddings, index, top_n=100):
    """
    1. Look up the user's embedding.
    2. Find the top_n nearest movies in embedding space.
    3. Remove movies the user already rated.
    """
    if user_id not in user_to_idx:
        return []

    user_idx = user_to_idx[user_id]
    user_vec = user_embeddings[user_idx].reshape(1, -1)

    # fetch extra candidates to account for filtering seen movies
    n_fetch = top_n + 200
    distances, indices = index.kneighbors(user_vec, n_neighbors=n_fetch)

    seen = set(ratings[ratings["user_id"] == user_id]["movie_id"])

    candidates = []
    for idx in indices[0]:
        mid = idx_to_movie[idx]
        if mid not in seen:
            candidates.append(mid)
        if len(candidates) >= top_n:
            break

    return candidates

if __name__ == "__main__":
    ratings = load_ratings()
    movies  = load_movies().set_index("movie_id")
    _, user_to_idx, movie_to_idx, idx_to_movie = build_feedback_matrix(ratings)

    user_embeddings = np.load("../data/user_embeddings.npy")
    item_embeddings = np.load("../data/item_embeddings.npy")

    print(f"User embeddings: {user_embeddings.shape}")
    print(f"Item embeddings: {item_embeddings.shape}")

    print("\nBuilding nearest-neighbor index...")
    index = build_index(item_embeddings)
    print("Index ready.")

    user_id = 1
    candidates = generate_candidates(
        user_id, user_to_idx, idx_to_movie, ratings,
        user_embeddings, item_embeddings, index, top_n=10
    )

    print(f"\nTop 10 collaborative filtering candidates for user {user_id}:")
    for mid in candidates:
        title  = movies.loc[mid, "title"]  if mid in movies.index else "?"
        genres = movies.loc[mid, "genres"] if mid in movies.index else "?"
        print(f"  {title}  ({genres})")
