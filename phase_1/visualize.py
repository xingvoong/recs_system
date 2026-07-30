import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data import load_movies, load_ratings
from matrix import build_feedback_matrix

item_embeddings = np.load("../data/item_embeddings.npy")
movies = load_movies()
ratings = load_ratings()
_, _, movie_to_idx, idx_to_movie = build_feedback_matrix(ratings)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(item_embeddings)
print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

GENRES = ["Animation", "Horror", "Romance", "Action", "Documentary", "Musical"]
COLORS = ["#1f77b4", "#d62728", "#e377c2", "#ff7f0e", "#2ca02c", "#9467bd"]

def get_idxs(genre):
    mask = movies["genre_list"].apply(lambda g: genre in g)
    ids = set(movies.loc[mask, "movie_id"].values)
    return [movie_to_idx[mid] for mid in ids if mid in movie_to_idx]

# ── Figure 1: all genres on one plot, bigger dots ─────────────────────────
fig, ax = plt.subplots(figsize=(13, 9))
ax.scatter(coords[:, 0], coords[:, 1], s=6, c="#cccccc", alpha=0.3, zorder=1)

for genre, color in zip(GENRES, COLORS):
    idxs = get_idxs(genre)
    ax.scatter(coords[idxs, 0], coords[idxs, 1],
               s=30, color=color, alpha=0.8, label=f"{genre} ({len(idxs)})", zorder=2)

ax.set_title("Movie Embeddings — WALS + PCA\nEach dot = one movie. Distance = audience overlap.", fontsize=13)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
ax.legend(fontsize=10, markerscale=1.5, title="Genre (movie count)")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("../data/embeddings_pca.png", dpi=150)
print("Saved embeddings_pca.png")

# ── Figure 2: one panel per genre so nothing is hidden ────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Genre Clusters in Embedding Space\n(grey = all movies, colour = that genre)",
             fontsize=13, fontweight="bold")

for ax, genre, color in zip(axes.flat, GENRES, COLORS):
    idxs = get_idxs(genre)
    ax.scatter(coords[:, 0], coords[:, 1], s=4, c="#dddddd", alpha=0.3, zorder=1)
    ax.scatter(coords[idxs, 0], coords[idxs, 1],
               s=25, color=color, alpha=0.85, zorder=2)
    ax.set_title(f"{genre}  ({len(idxs)} movies)", fontsize=11, color=color, fontweight="bold")
    ax.set_xlabel("PC1", fontsize=9)
    ax.set_ylabel("PC2", fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig("../data/embeddings_pca_grid.png", dpi=150)
print("Saved embeddings_pca_grid.png")
