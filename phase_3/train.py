import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle
import sys
sys.path.append("../phase_1")
sys.path.append("../phase_2")
from data import load_ratings, load_movies
from features import build_user_features, build_item_features, build_training_pair

# Label: 1 if user rated movie >= 4 (liked it), 0 otherwise
POSITIVE_THRESHOLD = 4
NEGATIVES_PER_POSITIVE = 4  # negative sampling ratio

def build_dataset(ratings, user_features, item_features, seed=42):
    """
    Positives: (user, movie) pairs where rating >= threshold
    Negatives: for each positive, sample N movies the user never rated

    This is negative sampling — we can't train on all unrated pairs
    (that's 95% of the matrix). Instead we sample a subset of negatives.
    Hard negatives would be movies the model scores high but are wrong;
    here we use random negatives as a baseline.
    """
    rng = np.random.default_rng(seed)
    all_movie_ids = list(item_features.keys())

    X, y = [], []

    for user_id, group in ratings.groupby("user_id"):
        if user_id not in user_features:
            continue

        positives = group[group["rating"] >= POSITIVE_THRESHOLD]["movie_id"].tolist()
        seen      = set(group["movie_id"].tolist())

        if not positives:
            continue

        # positive examples
        for mid in positives:
            if mid in item_features:
                X.append(build_training_pair(user_id, mid, user_features, item_features))
                y.append(1)

        # negative examples — random unseen movies
        unseen = [m for m in all_movie_ids if m not in seen]
        n_neg  = min(len(positives) * NEGATIVES_PER_POSITIVE, len(unseen))
        neg_sample = rng.choice(unseen, size=n_neg, replace=False)

        for mid in neg_sample:
            X.append(build_training_pair(user_id, mid, user_features, item_features))
            y.append(0)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    ratings = load_ratings()
    movies  = load_movies()

    print("Building features...")
    user_features = build_user_features(ratings, movies)
    item_features = build_item_features(movies, ratings)

    print("Building training dataset with negative sampling...")
    X, y = build_dataset(ratings, user_features, item_features)
    print(f"  Total examples : {len(X):,}")
    print(f"  Positives      : {y.sum():,}  ({100*y.mean():.1f}%)")
    print(f"  Negatives      : {(1-y).sum():,}  ({100*(1-y).mean():.1f}%)")
    print(f"  Feature dims   : {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,}   Test: {len(X_test):,}")

    print("\nTraining DNN scorer...")
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=30,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    print(f"\nTest AUC: {auc:.4f}")

    with open("../data/scorer.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to data/scorer.pkl")
