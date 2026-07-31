import numpy as np
import sys
sys.path.append("../phase_1")
sys.path.append("../phase_2")
sys.path.append("../phase_3")
from data import load_movies, load_ratings

def extract_year(title):
    try:
        return int(title.strip()[-5:-1])
    except:
        return 1990

def freshness_weight(year, min_year=1919, max_year=2000, strength=0.3):
    """
    Boost newer movies. Returns a multiplier in [1-strength, 1+strength].
    strength=0.3 means newest films get 1.3x, oldest get 0.7x.
    Keeps older classics reachable — just slightly deprioritised.
    """
    norm = (year - min_year) / (max_year - min_year)  # 0.0 → 1.0
    return 1.0 + strength * (2 * norm - 1)             # 0.7 → 1.3

def filter_seen(ranked, seen_ids):
    """Remove movies the user has already rated."""
    return [(mid, score) for mid, score in ranked if mid not in seen_ids]

def apply_freshness(ranked, movies_df):
    """Multiply each score by a freshness weight based on release year."""
    year_map = {
        row["movie_id"]: extract_year(row["title"])
        for _, row in movies_df.iterrows()
    }
    boosted = []
    for mid, score in ranked:
        year   = year_map.get(mid, 1990)
        weight = freshness_weight(year)
        boosted.append((mid, score * weight))

    return sorted(boosted, key=lambda x: x[1], reverse=True)

def enforce_diversity(ranked, movies_df, top_n=10, min_genres=3):
    """
    Pick top_n movies ensuring at least min_genres distinct genres are represented.
    Greedy: take the highest-scored movie, then keep adding until
    diversity target is met or we run out of candidates.
    """
    genre_map = {
        row["movie_id"]: set(row["genre_list"])
        for _, row in movies_df.iterrows()
    }

    selected      = []
    genres_seen   = set()

    # first pass: fill greedily
    for mid, score in ranked:
        if len(selected) >= top_n:
            break
        selected.append((mid, score))
        genres_seen.update(genre_map.get(mid, set()))

    # if diversity target not met, swap in candidates that add new genres
    if len(genres_seen) < min_genres:
        remaining = [(mid, score) for mid, score in ranked if (mid, score) not in selected]
        for mid, score in remaining:
            new_genres = genre_map.get(mid, set()) - genres_seen
            if new_genres:
                # replace the lowest-scored item in selected
                selected.sort(key=lambda x: x[1])
                selected[0] = (mid, score)
                genres_seen.update(new_genres)
                selected.sort(key=lambda x: x[1], reverse=True)
            if len(genres_seen) >= min_genres:
                break

    return selected

def rerank(user_id, scored_candidates, ratings, movies_df, top_n=10):
    """
    Full re-ranking pipeline:
      1. Filter already-seen movies
      2. Apply freshness boost
      3. Enforce genre diversity
    """
    seen_ids = set(ratings[ratings["user_id"] == user_id]["movie_id"])

    after_filter    = filter_seen(scored_candidates, seen_ids)
    after_freshness = apply_freshness(after_filter, movies_df)
    after_diversity = enforce_diversity(after_freshness, movies_df, top_n=top_n)

    return after_diversity, {
        "after_filter":    len(after_filter),
        "after_freshness": len(after_freshness),
        "final":           len(after_diversity),
    }

if __name__ == "__main__":
    # verify freshness weights make sense
    print("=== Freshness weights by decade ===")
    for year in [1930, 1950, 1970, 1985, 1995, 2000]:
        w = freshness_weight(year)
        print(f"  {year}: {w:.3f}x")
