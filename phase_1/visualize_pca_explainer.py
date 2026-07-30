import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

rng = np.random.default_rng(42)

# Simulate correlated 2D data (action vs romance scores)
n = 80
action  = rng.uniform(0, 10, n)
romance = 10 - action + rng.normal(0, 1.5, n)
romance = np.clip(romance, 0, 10)
X = np.stack([action, romance], axis=1)

pca = PCA(n_components=2)
pca.fit(X)
center = X.mean(axis=0)
pc1 = pca.components_[0]
pc2 = pca.components_[1]
scale1 = np.sqrt(pca.explained_variance_[0]) * 2.5
scale2 = np.sqrt(pca.explained_variance_[1]) * 2.5

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("How PCA Works", fontsize=14, fontweight="bold", y=1.02)

# ── Panel 1: raw data ──────────────────────────────────────────────────────
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], s=30, color="steelblue", alpha=0.7)
ax.set_title("1. Raw data\n(each dot = one movie, 2 features)", fontsize=10)
ax.set_xlabel("Action score")
ax.set_ylabel("Romance score")
ax.set_xlim(-1, 12); ax.set_ylim(-1, 12)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)
ax.annotate("movies spread\nalong a diagonal", xy=(7, 4), xytext=(2, 1),
            fontsize=8, color="grey",
            arrowprops=dict(arrowstyle="->", color="grey"))

# ── Panel 2: PCA finds directions ─────────────────────────────────────────
ax = axes[1]
ax.scatter(X[:, 0], X[:, 1], s=30, color="steelblue", alpha=0.7)

ax.annotate("", xy=center + scale1 * pc1, xytext=center - scale1 * pc1,
            arrowprops=dict(arrowstyle="<->", color="crimson", lw=2.5))
ax.annotate("", xy=center + scale2 * pc2, xytext=center - scale2 * pc2,
            arrowprops=dict(arrowstyle="<->", color="darkorange", lw=2.5))

ax.text(*(center + scale1 * pc1 + np.array([0.2, 0.2])),
        f"PC1\n({pca.explained_variance_ratio_[0]*100:.0f}% variance)",
        color="crimson", fontsize=8, fontweight="bold")
ax.text(*(center + scale2 * pc2 + np.array([0.2, 0.2])),
        f"PC2\n({pca.explained_variance_ratio_[1]*100:.0f}% variance)",
        color="darkorange", fontsize=8, fontweight="bold")

ax.set_title("2. PCA finds directions\nof maximum spread", fontsize=10)
ax.set_xlabel("Action score")
ax.set_ylabel("Romance score")
ax.set_xlim(-1, 12); ax.set_ylim(-1, 12)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

# ── Panel 3: projected coordinates ────────────────────────────────────────
X_transformed = pca.transform(X)
ax = axes[2]
ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
           s=30, color="steelblue", alpha=0.7)

ax.axhline(0, color="crimson",    lw=1.5, alpha=0.6, label="PC1 axis")
ax.axvline(0, color="darkorange", lw=1.5, alpha=0.6, label="PC2 axis")

ax.set_title("3. Movies projected onto PC axes\n(new coordinates)", fontsize=10)
ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]*100:.0f}% of spread)")
ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]*100:.0f}% of spread)")
ax.grid(True, alpha=0.2)
ax.legend(fontsize=8)

# annotate a single point to show the transformation
i = 10
ax.scatter(*X_transformed[i], s=80, color="red", zorder=5)
ax.annotate(f"({X_transformed[i,0]:.1f}, {X_transformed[i,1]:.1f})",
            xy=X_transformed[i], xytext=X_transformed[i] + np.array([0.3, 0.3]),
            fontsize=8, color="red")

plt.tight_layout()
plt.savefig("../data/pca_explainer.png", dpi=150, bbox_inches="tight")
print("Saved to data/pca_explainer.png")
