import numpy as np
import sys
sys.path.append("../phase_1")
from data import load_ratings, load_movies
from matrix import build_feedback_matrix
from content_based import build_movie_vectors, generate_candidates as cb_candidates
from collaborative import build_index, generate_candidates as cf_candidates

def pool_candidates(cb, cf):
    """
    Merge candidates from both generators.
    Preserve order: interleave so neither source dominates.
    Deduplicate while keeping first occurrence.
    """
    seen = set()
    merged = []
    for mid in cb + cf:
        if mid not in seen:
            seen.add(mid)
            merged.append(mid)
    return merged

if __name__ == "__main__":
    ratings = load_ratings()
    movies  = load_movies()
    _, user_to_idx, movie_to_idx, idx_to_movie = build_feedback_matrix(ratings)

    movie_vectors   = build_movie_vectors(movies)
    user_embeddings = np.load("../data/user_embeddings.npy")
    item_embeddings = np.load("../data/item_embeddings.npy")
    index           = build_index(item_embeddings)

    movie_info = movies.set_index("movie_id")

    for user_id in [1, 100, 500]:
        cb = cb_candidates(user_id, ratings, movies, movie_vectors, top_n=100)
        cf = cf_candidates(
            user_id, user_to_idx, idx_to_movie, ratings,
            user_embeddings, item_embeddings, index, top_n=100
        )
        pool = pool_candidates(cb, cf)

        print(f"\nUser {user_id}")
        print(f"  Content-based:  {len(cb)} candidates")
        print(f"  Collaborative:  {len(cf)} candidates")
        print(f"  Combined pool:  {len(pool)} candidates  "
              f"({len(cb) + len(cf) - len(pool)} duplicates removed)")
        print(f"  First 5 from pool:")
        for mid in pool[:5]:
            print(f"    {movie_info.loc[mid, 'title']}  "
                  f"({movie_info.loc[mid, 'genres']})")
