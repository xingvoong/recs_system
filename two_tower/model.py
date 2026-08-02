import numpy as np

"""
Two-Tower model built in pure numpy.

Each tower is a small MLP:
  input (20 dims) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16) → L2 normalize

The output is a unit-norm embedding vector. Cosine similarity between
a user embedding and an item embedding is then just their dot product.

No TensorFlow or PyTorch — numpy only. This makes every operation explicit:
you can see exactly what a forward pass does and where gradients come from.
"""

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    """Gradient of ReLU — 1 where x > 0, else 0."""
    return (x > 0).astype(np.float32)

def l2_normalize(x):
    """Scale each vector to unit norm."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + 1e-8)


class Tower:
    """
    A single MLP tower: input_dim → 64 → 32 → embedding_dim.
    Used for both the user tower and the item tower.

    Weights are stored as a list of (W, b) pairs — one per layer.
    """

    def __init__(self, input_dim=20, hidden_dims=(64, 32), embedding_dim=16, seed=None):
        rng = np.random.default_rng(seed)
        self.embedding_dim = embedding_dim

        # build weight matrices with He initialisation (good default for ReLU)
        layer_dims = [input_dim] + list(hidden_dims) + [embedding_dim]
        self.weights = []
        for in_d, out_d in zip(layer_dims[:-1], layer_dims[1:]):
            W = rng.standard_normal((in_d, out_d)).astype(np.float32) * np.sqrt(2.0 / in_d)
            b = np.zeros(out_d, dtype=np.float32)
            self.weights.append((W, b))

    def forward(self, X):
        """
        Forward pass through the tower.
        X: (batch, input_dim)
        Returns: (batch, embedding_dim) — L2-normalised embeddings
        Also stores intermediate activations needed for backprop.
        """
        self._cache = []  # store (pre-activation, post-activation) per layer
        out = X.astype(np.float32)

        for i, (W, b) in enumerate(self.weights):
            pre = out @ W + b                                    # linear
            post = relu(pre) if i < len(self.weights) - 1 else pre  # ReLU on hidden, linear on last
            self._cache.append((out, pre, post))
            out = post

        return l2_normalize(out)  # unit-norm output embedding

    def backward(self, X, grad_embedding, lr=1e-3):
        """
        Backprop through the tower and update weights.
        grad_embedding: gradient flowing back from the loss into the embedding (batch, embedding_dim)
        """
        # undo L2 normalisation gradient
        out_unnorm = self._cache[-1][2]
        norm = np.linalg.norm(out_unnorm, axis=-1, keepdims=True) + 1e-8
        # Jacobian of L2 norm: I/norm - x xT / norm^3
        dot = (out_unnorm * grad_embedding).sum(axis=-1, keepdims=True)
        grad = (grad_embedding - out_unnorm * dot / norm**2) / norm

        # backprop through each layer in reverse
        for i in reversed(range(len(self.weights))):
            input_i, pre_i, post_i = self._cache[i]
            W, b = self.weights[i]

            # gradient through ReLU (last layer has no ReLU)
            if i < len(self.weights) - 1:
                grad = grad * relu_grad(pre_i)

            dW = input_i.T @ grad / len(input_i)
            db = grad.mean(axis=0)
            grad = grad @ W.T

            # gradient descent step
            self.weights[i] = (W - lr * dW, b - lr * db)

    def save(self, path):
        arrays = {}
        for i, (W, b) in enumerate(self.weights):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)

    def load(self, path):
        data = np.load(path)
        n_layers = len(self.weights)
        self.weights = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n_layers)]


class TwoTowerModel:
    """
    Wraps the user tower and item tower together.
    Training minimises contrastive loss:
      - positive pairs (user, liked item) → push embeddings together
      - negative pairs (user, random item) → push embeddings apart
    """

    def __init__(self, input_dim=20, hidden_dims=(64, 32), embedding_dim=16):
        self.user_tower = Tower(input_dim, hidden_dims, embedding_dim, seed=42)
        self.item_tower = Tower(input_dim, hidden_dims, embedding_dim, seed=99)

    def embed_users(self, X):
        return self.user_tower.forward(X)

    def embed_items(self, X):
        return self.item_tower.forward(X)

    def similarity(self, user_X, item_X):
        """Cosine similarity between user and item embeddings."""
        u = self.embed_users(user_X)
        v = self.embed_items(item_X)
        return (u * v).sum(axis=-1)   # dot product of unit vectors = cosine sim


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    N_USERS = 50
    N_ITEMS = 50

    user_X = rng.standard_normal((N_USERS, 20)).astype(np.float32)
    item_X = rng.standard_normal((N_ITEMS, 20)).astype(np.float32)

    model = TwoTowerModel(input_dim=20, hidden_dims=(64, 32), embedding_dim=16)

    u_emb = model.embed_users(user_X)
    i_emb = model.embed_items(item_X)

    print("User embeddings:", u_emb.shape, "  unit-norm:", np.allclose(np.linalg.norm(u_emb, axis=1), 1.0, atol=1e-5))
    print("Item embeddings:", i_emb.shape)

    # full similarity matrix: every user vs every item
    sim_matrix = u_emb @ i_emb.T   # (N_USERS, N_ITEMS)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Two-Tower Model — Untrained Baseline", fontsize=13, fontweight="bold")

    # ── Plot 1: Ranked similarity curves per user ──────────────────────────
    ax = axes[0]
    for i in range(N_USERS):
        ranked = np.sort(sim_matrix[i])[::-1]
        ax.plot(ranked, color="#4C72B0", alpha=0.15, lw=0.8)
    mean_ranked = np.sort(sim_matrix, axis=1)[:, ::-1].mean(axis=0)
    ax.plot(mean_ranked, color="crimson", lw=2, label="mean across users")
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.4)
    ax.set_title("Ranked Item Similarities per User\n(each line = one user)", fontsize=10)
    ax.set_xlabel("Item rank (1 = most similar)")
    ax.set_ylabel("Cosine similarity")
    ax.legend(fontsize=9)
    ax.text(0.5, -0.15,
            "Untrained: scores are flat — no item stands out.\nAfter training: rank 1 should spike toward +1 for liked items.",
            transform=ax.transAxes, ha="center", fontsize=8, color="grey")

    # ── Plot 2: Similarity distribution ────────────────────────────────────
    ax = axes[1]
    all_sims = sim_matrix.flatten()
    ax.hist(all_sims, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(all_sims.mean(), color="crimson", lw=2, label=f"mean = {all_sims.mean():.3f}")
    ax.set_title("Distribution of Similarity Scores\n(all user-item pairs)", fontsize=10)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.text(0.5, -0.15,
            "Untrained: scores centred near 0.\nAfter training: positives shift right, negatives shift left.",
            transform=ax.transAxes, ha="center", fontsize=8, color="grey")

    # ── Plot 3: PCA of embeddings in shared space ───────────────────────────
    ax = axes[2]
    all_emb = np.vstack([u_emb, i_emb])   # (100, 16)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_emb)

    ax.scatter(coords[:N_USERS, 0], coords[:N_USERS, 1],
               s=40, color="#4C72B0", alpha=0.8, label="Users", zorder=2)
    ax.scatter(coords[N_USERS:, 0], coords[N_USERS:, 1],
               s=40, color="#DD8452", alpha=0.8, label="Items", marker="^", zorder=2)
    ax.set_title("User & Item Embeddings (PCA 2D)\nShared embedding space", fontsize=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% variance)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.text(0.5, -0.15,
            "Untrained: users and items are randomly mixed.\nAfter training: liked user-item pairs should cluster together.",
            transform=ax.transAxes, ha="center", fontsize=8, color="grey")

    plt.tight_layout()
    plt.savefig("images/two_tower_untrained.png", dpi=150, bbox_inches="tight")
    print("\nSaved visualization to data/two_tower_untrained.png")
