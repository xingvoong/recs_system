import numpy as np
import pickle
import sys
sys.path.append("../phase_1")
sys.path.append("../phase_2")
sys.path.append("../phase_3")
from data import load_ratings, load_movies
from matrix import build_feedback_matrix
from content_based import build_movie_vectors, generate_candidates as cb_candidates
from collaborative import build_index, generate_candidates as cf_candidates
from pool import pool_candidates
from features import build_user_features, build_item_features
from score import score_candidates
from rerank import rerank

def recommend(user_id, ratings, movies, model,
              user_features, item_features,
              movie_vectors, user_embeddings, item_embeddings, index,
              user_to_idx, idx_to_movie, top_n=10):

    # Stage 1 — candidate generation
    cb   = cb_candidates(user_id, ratings, movies, movie_vectors, top_n=100)
    cf   = cf_candidates(user_id, user_to_idx, idx_to_movie, ratings,
                         user_embeddings, item_embeddings, index, top_n=100)
    pool = pool_candidates(cb, cf)

    # Stage 2 — scoring
    scored = score_candidates(user_id, pool, model, user_features, item_features)

    # Stage 3 — re-ranking
    final, stats = rerank(user_id, scored, ratings, movies, top_n=top_n)

    return final, {
        "candidates_generated": len(pool),
        "after_scoring":        len(scored),
        **stats,
    }

if __name__ == "__main__":
    print("Loading data...")
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

    print()
    for user_id in [1, 100, 500]:
        final, stats = recommend(
            user_id, ratings, movies, model,
            user_features, item_features,
            movie_vectors, user_embeddings, item_embeddings, index,
            user_to_idx, idx_to_movie,
        )

        genres_in_top10 = set()
        for mid, _ in final:
            if mid in movie_info.index:
                for g in movie_info.loc[mid, "genre_list"]:
                    genres_in_top10.add(g)

        print(f"User {user_id}  |  pipeline: {stats['candidates_generated']} candidates "
              f"→ {stats['after_scoring']} scored → {stats['final']} final")
        print(f"  Genres in top 10: {sorted(genres_in_top10)}")
        print()
        for rank, (mid, score) in enumerate(final, 1):
            title  = movie_info.loc[mid, "title"]  if mid in movie_info.index else "?"
            year   = title[-5:-1] if len(title) > 6 else "?"
            genres = movie_info.loc[mid, "genres"] if mid in movie_info.index else "?"
            print(f"  {rank:2}. [{score:.3f}]  {title}")
            print(f"        {genres}")
        print()
