import numpy as np
import sys
sys.path.append("../phase_1")
sys.path.append("../phase_2")
sys.path.append("../phase_3")
from data import load_ratings, load_movies
from features import build_user_features, build_item_features, get_genre_matrix
from matrix import build_feedback_matrix
from collaborative import build_index as wals_build_index, generate_candidates as wals_candidates
from model import TwoTowerModel
from candidates import build_item_index, generate_candidates, generate_candidates_cold

"""
Cold-start comparison: WALS vs Two-Tower.

WALS stores one embedding vector per user, learned during training.
Ask for a user it has never seen → no vector → no recommendations.

Two-tower learns a function: features → embedding.
Ask for a new user → compute their feature vector → run through user tower → recommendations.
No historical ratings required.

This file runs three scenarios:
  1. Existing user   — both systems work, compare quality
  2. Cold-start user — WALS fails, two-tower handles it
  3. Minimal-data user (3 ratings) — shows two-tower degrades gracefully
"""


def wals_recommend(user_id, ratings, user_to_idx, idx_to_movie,
                   user_embeddings, item_embeddings, wals_index, movies, top_n=10):
    """Try WALS. Returns (list of (title, genres), error_message)."""
    if user_id not in user_to_idx:
        return [], f"WALS has no embedding for user {user_id} — cold-start fails."
    cands = wals_candidates(
        user_id, user_to_idx, idx_to_movie, ratings,
        user_embeddings, item_embeddings, wals_index, top_n=top_n
    )
    results = []
    for mid in cands:
        title  = movies.loc[mid, "title"]  if mid in movies.index else "?"
        genres = movies.loc[mid, "genres"] if mid in movies.index else "?"
        results.append((title, genres))
    return results, None


def two_tower_recommend(user_feat, seen_ids, user_tower, item_ids, tt_index, movies, top_n=10):
    """Two-tower recommend given a feature vector."""
    cands = generate_candidates_cold(
        user_feat, seen_ids, user_tower, item_ids, tt_index, top_n=top_n
    )
    results = []
    for mid in cands:
        title  = movies.loc[mid, "title"]  if mid in movies.index else "?"
        genres = movies.loc[mid, "genres"] if mid in movies.index else "?"
        results.append((title, genres))
    return results


def print_comparison(label, wals_results, wals_err, tt_results):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  {'WALS':35s}  {'Two-Tower'}")
    print(f"  {'─'*33}  {'─'*33}")

    if wals_err:
        wals_lines = [wals_err] + [""] * 9
    else:
        wals_lines = [f"{t[:30]}  ({g[:15]})" for t, g in wals_results]

    tt_lines = [f"{t[:30]}  ({g[:15]})" for t, g in tt_results]

    for i in range(max(len(wals_lines), len(tt_lines))):
        w = wals_lines[i] if i < len(wals_lines) else ""
        t = tt_lines[i]   if i < len(tt_lines)   else ""
        print(f"  {w:35s}  {t}")


if __name__ == "__main__":
    print("Loading data...")
    ratings = load_ratings()
    movies  = load_movies().set_index("movie_id")

    print("Building features...")
    user_features = build_user_features(ratings, movies.reset_index())
    item_features = build_item_features(movies.reset_index(), ratings)

    # ── Load WALS ──────────────────────────────────────────────────────────
    print("Loading WALS...")
    _, user_to_idx, _, idx_to_movie = build_feedback_matrix(ratings)
    user_embeddings = np.load("../data/user_embeddings.npy")
    item_embeddings = np.load("../data/item_embeddings.npy")
    wals_index = wals_build_index(item_embeddings)

    # ── Load Two-Tower ─────────────────────────────────────────────────────
    print("Loading two-tower...")
    model = TwoTowerModel(input_dim=20, hidden_dims=(64, 32), embedding_dim=16)
    model.user_tower.load("../data/user_tower.npz")
    model.item_tower.load("../data/item_tower.npz")
    item_ids, tt_index = build_item_index(item_features, model.item_tower)

    # genre order for building cold-start feature vectors
    get_genre_matrix(movies.reset_index())
    import features as feat_module
    genre_order = feat_module.GENRES

    def make_cold_feat(avg_rating, n_ratings, genre_weights):
        """Build a 20-dim feature vector for a user with declared genre preferences."""
        gp = np.zeros(len(genre_order), dtype=np.float32)
        for g, w in genre_weights.items():
            if g in genre_order:
                gp[genre_order.index(g)] = w
        return np.concatenate([
            [avg_rating / 5.0, np.log1p(n_ratings) / 10.0],
            gp,
        ]).astype(np.float32)

    print("\n" + "═" * 62)
    print("  COLD-START COMPARISON: WALS vs Two-Tower")
    print("═" * 62)

    # ── Scenario 1: Existing user with full history ────────────────────────
    # User 2299 — lots of Action/Sci-Fi ratings (Star Wars, Terminator, etc.)
    user_id = 2299
    seen_2299 = set(ratings[ratings["user_id"] == user_id]["movie_id"])
    u_feat_2299 = user_features[user_id]

    w_res, w_err = wals_recommend(
        user_id, ratings, user_to_idx, idx_to_movie,
        user_embeddings, item_embeddings, wals_index, movies
    )
    tt_res = two_tower_recommend(
        u_feat_2299, seen_2299, model.user_tower, item_ids, tt_index, movies
    )
    print_comparison(
        f"Scenario 1 — Existing user {user_id} (Star Wars / Terminator fan, {len(seen_2299)} ratings)",
        w_res, w_err, tt_res
    )

    # ── Scenario 2: Brand-new user — zero ratings ─────────────────────────
    NEW_USER_ID = 999999  # not in the dataset

    cold_feat = make_cold_feat(
        avg_rating=4.0,
        n_ratings=0,
        genre_weights={"Action": 1.0, "Sci-Fi": 1.0, "Thriller": 0.5}
    )

    w_res2, w_err2 = wals_recommend(
        NEW_USER_ID, ratings, user_to_idx, idx_to_movie,
        user_embeddings, item_embeddings, wals_index, movies
    )
    tt_res2 = two_tower_recommend(
        cold_feat, set(), model.user_tower, item_ids, tt_index, movies
    )
    print_comparison(
        "Scenario 2 — Brand-new user (Action / Sci-Fi fan, 0 ratings)",
        w_res2, w_err2, tt_res2
    )

    # ── Scenario 3: New user with 3 ratings ───────────────────────────────
    # We build their feature vector from 3 known likes, then ask both systems.
    # WALS still fails (not in training set). Two-tower uses the feature vector.
    liked_3 = ["Star Wars: Episode IV - A New Hope (1977)",
                "The Matrix (1999)",
                "Terminator 2: Judgment Day (1991)"]

    minimal_feat = make_cold_feat(
        avg_rating=4.5,
        n_ratings=3,
        genre_weights={"Action": 1.0, "Sci-Fi": 1.0, "Thriller": 0.3}
    )

    w_res3, w_err3 = wals_recommend(
        NEW_USER_ID, ratings, user_to_idx, idx_to_movie,
        user_embeddings, item_embeddings, wals_index, movies
    )
    tt_res3 = two_tower_recommend(
        minimal_feat, set(), model.user_tower, item_ids, tt_index, movies
    )
    print_comparison(
        f"Scenario 3 — New user with 3 declared likes\n  ({', '.join(liked_3[:2])}...)",
        w_res3, w_err3, tt_res3
    )

    print(f"\n{'═'*62}")
    print("  Summary")
    print(f"{'═'*62}")
    print("  WALS:       fast, accurate for known users — fails completely for new ones")
    print("  Two-Tower:  works for any user with a feature vector, even zero ratings")
    print("  Trade-off:  two-tower needs feature engineering; WALS only needs interactions")
