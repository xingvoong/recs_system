import numpy as np
from data import load_movies
from matrix import build_feedback_matrix
from data import load_ratings

item_embeddings = np.load("../data/item_embeddings.npy")
movies = load_movies().set_index("movie_id")
ratings = load_ratings()
_, _, movie_to_idx, idx_to_movie = build_feedback_matrix(ratings)

def get_idx(movie_id):
    return movie_to_idx[movie_id]

def top_k(scores, k=5):
    return np.argsort(scores)[::-1][:k]

def cosine_sim(query, matrix):
    query_norm = query / (np.linalg.norm(query) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    return (matrix / norms) @ query_norm

def dot_product_sim(query, matrix):
    return matrix @ query

def euclidean_sim(query, matrix):
    # negate so higher = more similar
    return -np.linalg.norm(matrix - query, axis=1)

def show_similar(movie_id, k=5):
    title = movies.loc[movie_id, "title"] if movie_id in movies.index else str(movie_id)
    query = item_embeddings[get_idx(movie_id)]

    cosine_scores   = cosine_sim(query, item_embeddings)
    dot_scores      = dot_product_sim(query, item_embeddings)
    euclidean_scores = euclidean_sim(query, item_embeddings)

    methods = [
        ("Cosine",     cosine_scores),
        ("Dot product", dot_scores),
        ("Euclidean",  euclidean_scores),
    ]

    print(f"\nQuery: {title}")
    print(f"Genres: {movies.loc[movie_id, 'genres'] if movie_id in movies.index else '?'}")
    print("=" * 60)

    for method_name, scores in methods:
        top = top_k(scores, k + 1)  # +1 to skip the query itself
        top = [i for i in top if idx_to_movie[i] != movie_id][:k]
        print(f"\n{method_name}:")
        for rank, idx in enumerate(top, 1):
            mid = idx_to_movie[idx]
            t = movies.loc[mid, "title"] if mid in movies.index else str(mid)
            g = movies.loc[mid, "genres"] if mid in movies.index else ""
            print(f"  {rank}. {t}")
            print(f"     {g}  (score: {scores[idx]:.4f})")

# Toy Story (1995) — movie_id 1
show_similar(1)

# The Silence of the Lambs (1991) — movie_id 1617
show_similar(1617)
