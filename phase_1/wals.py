import numpy as np
import implicit
from data import load_ratings, load_movies
from matrix import build_feedback_matrix

# WALS treats values as confidence weights, not raw ratings.
# A common formula: confidence = 1 + alpha * rating
# alpha scales how much a high rating boosts confidence vs. a low one.
ALPHA = 10
FACTORS = 32   # embedding dimension
ITERATIONS = 20

def ratings_to_confidence(matrix, alpha=ALPHA):
    # multiply in-place on a copy — keeps sparsity
    confident = matrix.copy().astype(np.float32)
    confident.data = 1 + alpha * confident.data
    return confident

if __name__ == "__main__":
    ratings = load_ratings()
    matrix, user_to_idx, movie_to_idx, idx_to_movie = build_feedback_matrix(ratings)

    confidence = ratings_to_confidence(matrix)

    # implicit expects item x user (transposed)
    item_user = confidence.T.tocsr()

    model = implicit.als.AlternatingLeastSquares(
        factors=FACTORS,
        iterations=ITERATIONS,
        regularization=0.01,
        random_state=42,
    )

    print(f"Training WALS — {FACTORS} factors, {ITERATIONS} iterations...")
    model.fit(item_user)

    # implicit's naming is counterintuitive when you pass item_user:
    # user_factors = row-side factors = movies (3706, 32)
    # item_factors = col-side factors = users  (6040, 32)
    item_embeddings = model.user_factors   # (n_movies, factors)
    user_embeddings = model.item_factors   # (n_users,  factors)

    print(f"\nUser embeddings: {user_embeddings.shape}")
    print(f"Item embeddings: {item_embeddings.shape}")
    print(f"\nSample user embedding (user index 0):\n  {user_embeddings[0].round(4)}")

    # quick sanity check — find top 5 movies for user 0
    user_vec = user_embeddings[0]
    scores = item_embeddings @ user_vec
    top5_idx = np.argsort(scores)[::-1][:5]

    movies = load_movies().set_index("movie_id")

    print("\nTop 5 recommendations for user index 0:")
    for rank, idx in enumerate(top5_idx, 1):
        mid = idx_to_movie[idx]
        title = movies.loc[mid, "title"] if mid in movies.index else "Unknown"
        print(f"  {rank}. [{mid}] {title}  (score: {scores[idx]:.4f})")

    np.save("../data/user_embeddings.npy", user_embeddings)
    np.save("../data/item_embeddings.npy", item_embeddings)
    print("\nEmbeddings saved to data/")
