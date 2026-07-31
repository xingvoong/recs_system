import numpy as np
import pickle
import sys
sys.path.append("../phase_1")
sys.path.append("../phase_2")
from data import load_ratings, load_movies
from matrix import build_feedback_matrix
from content_based import build_movie_vectors, generate_candidates as cb_candidates
from collaborative import build_index, generate_candidates as cf_candidates
from pool import pool_candidates
from features import build_user_features, build_item_features

def score_candidates(user_id, candidates, model, user_features, item_features):
    """
    Score each candidate with the DNN and return ranked list.
    Input vector = user features (20 dims) + item features (20 dims) = 40 dims.
    """
    if not candidates:
        return []

    u = user_features.get(user_id, np.zeros(20, dtype=np.float32))
    X = np.array([
        np.concatenate([u, item_features.get(mid, np.zeros(20, dtype=np.float32))])
        for mid in candidates
    ])

    scores = model.predict_proba(X)[:, 1]  # P(liked)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked  # [(movie_id, score), ...]

if __name__ == "__main__":
    ratings = load_ratings()
    movies  = load_movies()
    _, user_to_idx, movie_to_idx, idx_to_movie = build_feedback_matrix(ratings)
    movie_info = movies.set_index("movie_id")

    print("Loading model and features...")
    with open("../data/scorer.pkl", "rb") as f:
        model = pickle.load(f)

    user_features = build_user_features(ratings, movies)
    item_features = build_item_features(movies, ratings)

    print("Building candidate generators...")
    movie_vectors   = build_movie_vectors(movies)
    user_embeddings = np.load("../data/user_embeddings.npy")
    item_embeddings = np.load("../data/item_embeddings.npy")
    index           = build_index(item_embeddings)

    for user_id in [1, 100, 500]:
        cb   = cb_candidates(user_id, ratings, movies, movie_vectors, top_n=100)
        cf   = cf_candidates(user_id, user_to_idx, idx_to_movie, ratings,
                             user_embeddings, item_embeddings, index, top_n=100)
        pool = pool_candidates(cb, cf)

        ranked = score_candidates(user_id, pool, model, user_features, item_features)

        print(f"\n{'='*60}")
        print(f"User {user_id} — {len(pool)} candidates → scored → top 10")
        print(f"{'='*60}")
        for rank, (mid, score) in enumerate(ranked[:10], 1):
            title  = movie_info.loc[mid, "title"]  if mid in movie_info.index else "?"
            genres = movie_info.loc[mid, "genres"] if mid in movie_info.index else "?"
            print(f"  {rank:2}. [{score:.3f}]  {title}")
            print(f"        {genres}")
