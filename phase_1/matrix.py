import numpy as np
import scipy.sparse as sp
from data import load_ratings, load_movies

def build_feedback_matrix(ratings):
    """
    Build a sparse user x item matrix.
    Values are raw ratings (1–5).
    Row = user index, Col = movie index (0-based, contiguous).
    Returns the matrix plus index↔id mappings.
    """
    user_ids = sorted(ratings["user_id"].unique())
    movie_ids = sorted(ratings["movie_id"].unique())

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    rows = ratings["user_id"].map(user_to_idx).values
    cols = ratings["movie_id"].map(movie_to_idx).values
    data = ratings["rating"].values.astype(np.float32)

    matrix = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
    )

    idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}

    return matrix, user_to_idx, movie_to_idx, idx_to_movie

if __name__ == "__main__":
    ratings = load_ratings()
    matrix, user_to_idx, movie_to_idx, idx_to_movie = build_feedback_matrix(ratings)

    ratings_per_user = np.diff(matrix.indptr)
    ratings_per_movie = np.diff(matrix.tocsc().indptr)

    print("=== Feedback Matrix ===")
    print(f"  Shape:   {matrix.shape}  (users x movies)")
    print(f"  Stored:  {matrix.nnz:,} non-zero entries")
    print(f"  Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.2f}%")
    print()
    print("=== Ratings per User ===")
    print(f"  min={ratings_per_user.min()}  median={np.median(ratings_per_user):.0f}  max={ratings_per_user.max()}")
    print()
    print("=== Ratings per Movie ===")
    print(f"  min={ratings_per_movie.min()}  median={np.median(ratings_per_movie):.0f}  max={ratings_per_movie.max()}")
