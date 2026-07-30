import pandas as pd

DATA_DIR = "../data/ml-1m"

def load_ratings():
    return pd.read_csv(
        f"{DATA_DIR}/ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

def load_movies():
    movies = pd.read_csv(
        f"{DATA_DIR}/movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    movies["genre_list"] = movies["genres"].str.split("|")
    return movies

def load_users():
    return pd.read_csv(
        f"{DATA_DIR}/users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
    )

if __name__ == "__main__":
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()

    print("=== Ratings ===")
    print(f"  Shape:   {ratings.shape}")
    print(f"  Users:   {ratings['user_id'].nunique()}")
    print(f"  Movies:  {ratings['movie_id'].nunique()}")
    print(f"  Ratings: {len(ratings):,}")
    print(f"  Rating range: {ratings['rating'].min()} – {ratings['rating'].max()}")
    print()

    print("=== Movies ===")
    print(f"  Total movies: {len(movies)}")
    all_genres = movies["genre_list"].explode().unique()
    print(f"  Unique genres: {sorted(all_genres)}")
    print()

    print("=== Sparsity ===")
    n_users = ratings["user_id"].nunique()
    n_movies = ratings["movie_id"].nunique()
    possible = n_users * n_movies
    actual = len(ratings)
    print(f"  Matrix size: {n_users} x {n_movies} = {possible:,} possible entries")
    print(f"  Observed:    {actual:,} ({100 * actual / possible:.2f}% dense)")

    print()
    print("=== Sample ===")
    sample = ratings.merge(movies[["movie_id", "title", "genres"]], on="movie_id").head(5)
    print(sample[["user_id", "title", "rating", "genres"]].to_string(index=False))
