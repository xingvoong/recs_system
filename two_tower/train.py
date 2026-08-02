import numpy as np
import sys
sys.path.append("../phase_1")
sys.path.append("../phase_3")
from data import load_ratings, load_movies
from features import build_user_features, build_item_features
from model import TwoTowerModel

"""
Two-Tower training with triplet loss.

Each training step uses (user, positive_item, negative_item) triplets.
The loss pushes positive pairs closer and pulls negative pairs apart:

    loss = mean( max(0,  margin - sim(u, pos) + sim(u, neg)) )

This is the standard triplet margin loss. A triplet only contributes to the
loss when the negative is scored higher than the positive minus the margin —
so easy negatives (already separated) get zero gradient.
"""

POSITIVE_THRESHOLD   = 4     # rating >= 4 counts as a positive
NEGATIVES_PER_POSITIVE = 4   # same ratio used in phase_3
MARGIN               = 0.2   # triplet margin (kept for reference, not used by InfoNCE)
BATCH_SIZE           = 512
EPOCHS               = 20
LR                   = 1e-2
TEMPERATURE          = 0.07  # InfoNCE temperature — lower = sharper distribution


# ── Dataset ────────────────────────────────────────────────────────────────

def build_triplets(ratings, user_features, item_features, seed=42):
    """
    Returns three arrays, each row being one triplet:
      U  — (N, 20) user feature vectors
      P  — (N, 20) positive item feature vectors
      Ng — (N, 20) negative item feature vectors
    """
    rng = np.random.default_rng(seed)
    all_movie_ids = np.array(list(item_features.keys()))

    U, P, Ng = [], [], []

    for user_id, group in ratings.groupby("user_id"):
        if user_id not in user_features:
            continue

        positives = group[group["rating"] >= POSITIVE_THRESHOLD]["movie_id"].tolist()
        if not positives:
            continue

        seen      = set(group["movie_id"].tolist())
        u_feat    = user_features[user_id]

        # filter positives to movies we have features for
        pos_feats = [item_features[m] for m in positives if m in item_features]
        if not pos_feats:
            continue

        unseen_mask = ~np.isin(all_movie_ids, list(seen))
        unseen_ids  = all_movie_ids[unseen_mask]

        n_neg = len(pos_feats) * NEGATIVES_PER_POSITIVE
        if len(unseen_ids) < n_neg:
            continue

        neg_ids   = rng.choice(unseen_ids, size=n_neg, replace=False)
        neg_feats = np.array([item_features[m] for m in neg_ids], dtype=np.float32)

        # tile each positive with NEGATIVES_PER_POSITIVE negatives
        for k, pf in enumerate(pos_feats):
            for j in range(NEGATIVES_PER_POSITIVE):
                U.append(u_feat)
                P.append(pf)
                Ng.append(neg_feats[k * NEGATIVES_PER_POSITIVE + j])

    return (np.array(U,  dtype=np.float32),
            np.array(P,  dtype=np.float32),
            np.array(Ng, dtype=np.float32))


# ── Loss and gradients ─────────────────────────────────────────────────────

def triplet_loss_and_grad(u_emb, p_emb, n_emb, margin=MARGIN):
    """
    u_emb, p_emb, n_emb: (batch, emb_dim) unit-norm embeddings

    Returns:
      loss            — scalar
      grad_u          — (batch, emb_dim) gradient into user embedding
      grad_p          — (batch, emb_dim) gradient into positive item embedding
      grad_n          — (batch, emb_dim) gradient into negative item embedding
    """
    pos_sim = (u_emb * p_emb).sum(axis=1)   # (batch,)
    neg_sim = (u_emb * n_emb).sum(axis=1)   # (batch,)

    raw   = margin - pos_sim + neg_sim       # (batch,)
    active = raw > 0                         # only these triplets contribute

    loss = raw[active].mean() if active.any() else 0.0

    # gradient of loss w.r.t. similarity scores
    # normalise by batch size (not active count) for stable effective LR
    coeff = active.astype(np.float32) / len(active)            # (batch,)

    grad_u = -coeff[:, None] * p_emb + coeff[:, None] * n_emb  # (batch, d)
    grad_p = -coeff[:, None] * u_emb                            # (batch, d)
    grad_n =  coeff[:, None] * u_emb                            # (batch, d)

    return loss, grad_u, grad_p, grad_n


# ── Training loop ──────────────────────────────────────────────────────────

def train(model, U, P, Ng, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=0):
    rng = np.random.default_rng(seed)
    N   = len(U)
    history = []

    for epoch in range(epochs):
        idx = rng.permutation(N)
        U, P, Ng = U[idx], P[idx], Ng[idx]

        epoch_losses = []

        for start in range(0, N, batch_size):
            end = start + batch_size
            u_batch  = U[start:end]
            p_batch  = P[start:end]
            n_batch  = Ng[start:end]

            # forward — item tower processes pos and neg in one pass
            u_emb  = model.user_tower.forward(u_batch)
            all_items = np.vstack([p_batch, n_batch])
            all_i_emb = model.item_tower.forward(all_items)
            p_emb  = all_i_emb[:len(u_batch)]
            n_emb  = all_i_emb[len(u_batch):]

            loss, grad_u, grad_p, grad_n = triplet_loss_and_grad(u_emb, p_emb, n_emb)
            epoch_losses.append(loss)

            # backward — item tower gets stacked gradient for pos + neg
            grad_all_i = np.vstack([grad_p, grad_n])
            model.user_tower.backward(u_batch,    grad_u,     lr)
            model.item_tower.backward(all_items,  grad_all_i, lr)

        mean_loss = np.mean(epoch_losses)
        history.append(mean_loss)
        print(f"  Epoch {epoch + 1:2d}/{epochs}  loss = {mean_loss:.4f}")

    return history


# ── Metric collection (runs on any model state) ────────────────────────────

def collect_metrics(model, U, P, Ng, ratings, user_features, item_features,
                    sample_idx, sample_users, pca_uids, pca_iids):
    """
    Snapshot three metrics on the current model weights:
      - pos_sim / neg_sim : similarity scores on a sample of triplets
      - ranks             : where liked items rank among all items (per user)
      - coords            : 2-D PCA projection of user + item embeddings
    Uses fixed sample indices so before/after are directly comparable.
    """
    from sklearn.decomposition import PCA

    # (a) pos vs neg similarity
    u_s  = model.user_tower.forward(U[sample_idx])
    p_s  = model.item_tower.forward(P[sample_idx])
    n_s  = model.item_tower.forward(Ng[sample_idx])
    pos_sim = (u_s * p_s).sum(axis=1)
    neg_sim = (u_s * n_s).sum(axis=1)

    # (b) rank of liked items
    all_item_ids  = list(item_features.keys())
    all_item_vecs = np.array([item_features[m] for m in all_item_ids], dtype=np.float32)
    all_i_emb     = model.item_tower.forward(all_item_vecs)

    u_vecs_eval = np.array([user_features[uid] for uid in sample_users], dtype=np.float32)
    u_embs_eval = model.user_tower.forward(u_vecs_eval)
    sims_all    = u_embs_eval @ all_i_emb.T          # (n_users, n_items)

    pos_ratings = ratings[ratings["rating"] >= POSITIVE_THRESHOLD]
    ranks = []
    for i, uid in enumerate(sample_users):
        liked = pos_ratings[pos_ratings["user_id"] == uid]["movie_id"].tolist()
        if not liked:
            continue
        rank_order = np.argsort(-sims_all[i])
        rank_of = {all_item_ids[r]: pos + 1 for pos, r in enumerate(rank_order)}
        for mid in liked:
            if mid in rank_of:
                ranks.append(rank_of[mid])

    # (c) PCA
    N_PCA = len(pca_uids)
    u_emb_pca = model.user_tower.forward(
        np.array([user_features[uid] for uid in pca_uids], dtype=np.float32))
    i_emb_pca = model.item_tower.forward(
        np.array([item_features[mid] for mid in pca_iids], dtype=np.float32))
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(np.vstack([u_emb_pca, i_emb_pca]))
    var    = pca.explained_variance_ratio_

    return {
        "pos_sim": pos_sim,
        "neg_sim": neg_sim,
        "ranks":   np.array(ranks),
        "coords":  coords,
        "pca_var": var,
        "N_PCA":   N_PCA,
        "all_item_ids": all_item_ids,
    }


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Loading data...")
    ratings = load_ratings()
    movies  = load_movies()

    print("Building features...")
    user_features = build_user_features(ratings, movies)
    item_features = build_item_features(movies, ratings)
    print(f"  {len(user_features):,} users   {len(item_features):,} items")

    print("\nBuilding triplets...")
    U, P, Ng = build_triplets(ratings, user_features, item_features)
    print(f"  {len(U):,} triplets")

    # fix the sample indices now so before/after use identical data
    rng_eval     = np.random.default_rng(99)
    sample_idx   = rng_eval.choice(len(U), size=min(5000, len(U)), replace=False)
    sample_users = rng_eval.choice(
        list(user_features.keys()), size=min(300, len(user_features)), replace=False)
    all_item_ids = list(item_features.keys())
    N_PCA        = 200
    pca_uids     = rng_eval.choice(list(user_features.keys()), size=N_PCA, replace=False)
    pca_iids     = rng_eval.choice(all_item_ids,               size=N_PCA, replace=False)

    model = TwoTowerModel(input_dim=20, hidden_dims=(64, 32), embedding_dim=16)

    print("\nCollecting before-training metrics...")
    before = collect_metrics(model, U, P, Ng, ratings,
                             user_features, item_features,
                             sample_idx, sample_users, pca_uids, pca_iids)

    print("\nTraining...")
    history = train(model, U, P, Ng)

    model.user_tower.save("../data/user_tower")
    model.item_tower.save("../data/item_tower")
    print("\nWeights saved to data/user_tower.npz and data/item_tower.npz")

    print("\nCollecting after-training metrics...")
    after = collect_metrics(model, U, P, Ng, ratings,
                            user_features, item_features,
                            sample_idx, sample_users, pca_uids, pca_iids)

    # ── Before vs After comparison figure (3 rows × 2 cols) ────────────────
    print("\nGenerating comparison figure...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("Two-Tower Model — Before vs After Training", fontsize=14, fontweight="bold")

    col_labels = ["Before training  (random weights)", "After training  (10 epochs)"]
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=11, fontweight="bold", pad=12,
                               color="#333333")

    snapshots = [before, after]

    for col, snap in enumerate(snapshots):
        pos_sim = snap["pos_sim"]
        neg_sim = snap["neg_sim"]
        ranks   = snap["ranks"]
        coords  = snap["coords"]
        var     = snap["pca_var"]

        # ── Row 0: Positive vs Negative similarity distributions ────────────
        ax = axes[0, col]
        ax.hist(neg_sim, bins=50, alpha=0.7, color="#d62728", density=True,
                label=f"Negative  mean={neg_sim.mean():.2f}")
        ax.hist(pos_sim, bins=50, alpha=0.7, color="#2ca02c", density=True,
                label=f"Positive  mean={pos_sim.mean():.2f}")
        ax.axvline(neg_sim.mean(), color="#d62728", lw=2, linestyle="--")
        ax.axvline(pos_sim.mean(), color="#2ca02c", lw=2, linestyle="--")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        if col == 0:
            ax.text(-0.18, 0.5, "Pos vs Neg\nSimilarity", transform=ax.transAxes,
                    va="center", ha="center", fontsize=10, rotation=90, color="#555")

        # ── Row 1: Rank of liked items ───────────────────────────────────────
        ax = axes[1, col]
        ax.hist(ranks, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.axvline(np.median(ranks), color="black", lw=1.5, linestyle="--",
                   label=f"median = {np.median(ranks):.0f}")
        for k, col_k in [(10, "#2ca02c"), (50, "orange"), (100, "#d62728")]:
            r_at_k = (ranks <= k).mean()
            ax.axvline(k, color=col_k, lw=1.5, linestyle=":", alpha=0.9,
                       label=f"Recall@{k:3d} = {r_at_k:.2f}")
        ax.set_xlabel("Rank of liked item (lower = better)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        if col == 0:
            ax.text(-0.18, 0.5, "Rank of\nLiked Items", transform=ax.transAxes,
                    va="center", ha="center", fontsize=10, rotation=90, color="#555")

        # ── Row 2: PCA of embeddings ─────────────────────────────────────────
        ax = axes[2, col]
        ax.scatter(coords[:N_PCA, 0], coords[:N_PCA, 1],
                   s=20, color="#4C72B0", alpha=0.6, label="Users")
        ax.scatter(coords[N_PCA:, 0], coords[N_PCA:, 1],
                   s=20, color="#DD8452", alpha=0.6, label="Items", marker="^")
        ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        if col == 0:
            ax.text(-0.18, 0.5, "PCA of\nEmbeddings", transform=ax.transAxes,
                    va="center", ha="center", fontsize=10, rotation=90, color="#555")

    # row captions (shared across both columns)
    captions = [
        "Green = liked items, red = random negatives.\n"
        "A gap between the means means the model scores liked items higher.",

        "Each liked (user, movie) pair is ranked against all 3,883 items.\n"
        "Recall@K = % of liked items in the top K. Random chance ≈ K/3883.",

        "200 users (blue) and 200 items (orange) projected into 2D.\n"
        "Random weights → noise. Trained weights → geometric structure.",
    ]
    for row, caption in enumerate(captions):
        axes[row, 1].text(1.02, 0.5, caption,
                          transform=axes[row, 1].transAxes,
                          va="center", ha="left", fontsize=8, color="grey",
                          wrap=True)

    plt.tight_layout(rect=[0.05, 0, 0.88, 0.96])
    plt.savefig("images/two_tower_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved to data/two_tower_comparison.png")

    # ── Loss curve (separate small figure) ─────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(range(1, len(history) + 1), history, marker="o", color="#4C72B0", lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Triplet Loss")
    ax2.set_title("Two-Tower Training — Loss Curve")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/two_tower_loss.png", dpi=150)
    print("Saved to data/two_tower_loss.png")
