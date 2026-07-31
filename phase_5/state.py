"""
Global state loaded once at startup.
Every request reads from here — no repeated disk I/O or retraining.
"""
import pickle
import numpy as np
import sys
sys.path.append("../phase_1")
sys.path.append("../phase_2")
sys.path.append("../phase_3")

from data import load_ratings, load_movies
from matrix import build_feedback_matrix
from content_based import build_movie_vectors
from collaborative import build_index
from features import build_user_features, build_item_features

ratings         = None
movies          = None
movie_info      = None
user_to_idx     = None
idx_to_movie    = None
movie_to_idx    = None
movie_vectors   = None
user_embeddings = None
item_embeddings = None
cf_index        = None
user_features   = None
item_features   = None
scorer          = None

def load():
    global ratings, movies, movie_info
    global user_to_idx, idx_to_movie, movie_to_idx
    global movie_vectors, user_embeddings, item_embeddings, cf_index
    global user_features, item_features, scorer

    print("Loading ratings and movies...")
    ratings   = load_ratings()
    movies    = load_movies()
    movie_info = movies.set_index("movie_id")

    print("Building index mappings...")
    _, user_to_idx, movie_to_idx, idx_to_movie = build_feedback_matrix(ratings)

    print("Building content-based vectors...")
    movie_vectors = build_movie_vectors(movies)

    print("Loading WALS embeddings...")
    user_embeddings = np.load("../data/user_embeddings.npy")
    item_embeddings = np.load("../data/item_embeddings.npy")
    cf_index        = build_index(item_embeddings)

    print("Building user and item features...")
    user_features = build_user_features(ratings, movies)
    item_features = build_item_features(movies, ratings)

    print("Loading scorer model...")
    with open("../data/scorer.pkl", "rb") as f:
        scorer = pickle.load(f)

    print("Ready.")
