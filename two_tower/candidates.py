import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.append("../phase_1")
sys.path.append("../phase_3")
from data import load_ratings, load_movies
from features import build_user_features, build_item_features
from model import TwoTowerModel

"""
Candidate generation using the trained two-tower model.

WALS (phase_2/collaborative.py):
  - Looks up a pre-stored user embedding by index.
  - New user → no index → no candidates.

Two-tower:
  - Runs user features through the user tower at query time.
  - Any user with a feature vector gets an embedding, including new users.

The index is built once over all item embeddings (pre-computed and stored).
At query time, only the user embedding needs to be computed — one forward pass,
then a nearest-neighbor search.
"""


def build_item_index(item_features, item_tower):
    """
    Pre-compute embeddings for every item and build a KNN index over them.
    Returns (item_ids list, index).
    Called once at startup; results can be cached.
    """
    item_ids  = list(item_features.keys())
    item_vecs = np.array([item_features[m] for m in item_ids], dtype=np.float32)
    item_embs = item_tower.forward(item_vecs)          # (n_items, emb_dim)

    index = NearestNeighbors(metric="cosine", algorithm="brute")
    index.fit(item_embs)
    return item_ids, index


def generate_candidates(user_id, ratings, user_features, user_tower,
                        item_ids, index, top_n=100):
    """
    Generate top_n candidates for a known user.
    The user's feature vector is run through the user tower to get their embedding.
    Same interface as phase_2/collaborative.py — drop-in replacement.
    """
    if user_id not in user_features:
        return []

    u_feat = user_features[user_id].reshape(1, -1)
    u_emb  = user_tower.forward(u_feat)               # (1, emb_dim)

    n_fetch  = top_n + 200
    _, indices = index.kneighbors(u_emb, n_neighbors=min(n_fetch, len(item_ids)))

    seen = set(ratings[ratings["user_id"] == user_id]["movie_id"])

    candidates = []
    for idx in indices[0]:
        mid = item_ids[idx]
        if mid not in seen:
            candidates.append(mid)
        if len(candidates) >= top_n:
            break

    return candidates


def generate_candidates_cold(user_feat, seen_movie_ids, user_tower,
                             item_ids, index, top_n=100):
    """
    Generate candidates for a cold-start user — no ratings history, just features.
    user_feat: (20,) numpy array built from whatever is known about the user
    seen_movie_ids: set of movie IDs to exclude (can be empty for brand-new users)
    """
    u_emb = user_tower.forward(user_feat.reshape(1, -1))   # (1, emb_dim)

    n_fetch  = top_n + len(seen_movie_ids) + 10
    _, indices = index.kneighbors(u_emb, n_neighbors=min(n_fetch, len(item_ids)))

    candidates = []
    for idx in indices[0]:
        mid = item_ids[idx]
        if mid not in seen_movie_ids:
            candidates.append(mid)
        if len(candidates) >= top_n:
            break

    return candidates


if __name__ == "__main__":
    print("Loading data...")
    ratings = load_ratings()
    movies  = load_movies().set_index("movie_id")

    print("Building features...")
    user_features = build_user_features(ratings, movies.reset_index())
    item_features = build_item_features(movies.reset_index(), ratings)

    print("Loading trained towers...")
    model = TwoTowerModel(input_dim=20, hidden_dims=(64, 32), embedding_dim=16)
    model.user_tower.load("../data/user_tower.npz")
    model.item_tower.load("../data/item_tower.npz")

    print("Building item index...")
    item_ids, index = build_item_index(item_features, model.item_tower)
    print(f"  Index ready — {len(item_ids):,} items")

    # ── Known user ─────────────────────────────────────────────────────────
    user_id    = 1
    candidates = generate_candidates(
        user_id, ratings, user_features, model.user_tower,
        item_ids, index, top_n=10
    )

    print(f"\nTop 10 two-tower candidates for user {user_id}:")
    for mid in candidates:
        title  = movies.loc[mid, "title"]  if mid in movies.index else "?"
        genres = movies.loc[mid, "genres"] if mid in movies.index else "?"
        print(f"  {title}  ({genres})")

    # ── Cold-start user ─────────────────────────────────────────────────────
    # Simulate a new user: loves Action and Sci-Fi, no ratings yet.
    # The 20-dim feature layout is: [avg_rating/5, log1p(n)/10, genre_profile (18)]
    # Genre order must match what build_user_features used during training.
    # get_genre_matrix populates features.GENRES — use that as the source of truth.
    import features as feat_module
    feat_module.get_genre_matrix(movies.reset_index())   # populates feat_module.GENRES
    genre_order = feat_module.GENRES

    print(f"\nGenre order ({len(genre_order)} genres): {genre_order}")

    genre_profile = np.zeros(len(genre_order), dtype=np.float32)
    for g, weight in [("Action", 1.0), ("Sci-Fi", 1.0), ("Thriller", 0.5)]:
        if g in genre_order:
            genre_profile[genre_order.index(g)] = weight

    cold_feat = np.concatenate([
        [3.5 / 5.0,           # avg_rating normalised
         np.log1p(0) / 10.0], # activity = 0 ratings
        genre_profile,
    ]).astype(np.float32)

    cold_candidates = generate_candidates_cold(
        cold_feat, seen_movie_ids=set(),
        user_tower=model.user_tower,
        item_ids=item_ids, index=index, top_n=10
    )

    print("\nTop 10 candidates for cold-start user (Action / Sci-Fi fan, 0 ratings):")
    for mid in cold_candidates:
        title  = movies.loc[mid, "title"]  if mid in movies.index else "?"
        genres = movies.loc[mid, "genres"] if mid in movies.index else "?"
        print(f"  {title}  ({genres})")
