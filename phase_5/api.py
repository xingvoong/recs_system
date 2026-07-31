from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import sys
sys.path.append("../phase_2")
sys.path.append("../phase_3")
sys.path.append("../phase_4")

import state
from content_based import generate_candidates as cb_candidates
from collaborative import generate_candidates as cf_candidates
from pool import pool_candidates
from score import score_candidates
from rerank import rerank

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()   # runs once at startup
    yield

app = FastAPI(title="Movie Recommender", lifespan=lifespan)

@app.get("/recommend")
def recommend(user_id: int, top_n: int = 10):
    if user_id not in state.user_to_idx:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    cb   = cb_candidates(user_id, state.ratings, state.movies,
                         state.movie_vectors, top_n=100)
    cf   = cf_candidates(user_id, state.user_to_idx, state.idx_to_movie,
                         state.ratings, state.user_embeddings,
                         state.item_embeddings, state.cf_index, top_n=100)
    pool = pool_candidates(cb, cf)

    scored = score_candidates(user_id, pool, state.scorer,
                              state.user_features, state.item_features)

    final, rerank_stats = rerank(user_id, scored, state.ratings, state.movies, top_n=top_n)

    results = []
    for mid, score in final:
        if mid in state.movie_info.index:
            row = state.movie_info.loc[mid]
            results.append({
                "movie_id": int(mid),
                "title":    str(row["title"]),
                "genres":   str(row["genres"]),
                "score":    round(float(score), 4),
            })

    return {
        "user_id": user_id,
        "pipeline": {
            "candidates_generated": len(pool),
            "after_scoring":        len(scored),
            "after_filter":         rerank_stats["after_filter"],
            "final":                rerank_stats["final"],
        },
        "recommendations": results,
    }

@app.get("/similar")
def similar(movie_id: int, top_n: int = 10):
    if movie_id not in state.movie_to_idx:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")

    idx        = state.movie_to_idx[movie_id]
    query_vec  = state.item_embeddings[idx].reshape(1, -1)
    distances, indices = state.cf_index.kneighbors(query_vec, n_neighbors=top_n + 1)

    results = []
    for i in indices[0]:
        mid = state.idx_to_movie[i]
        if mid == movie_id:
            continue
        if mid in state.movie_info.index:
            row = state.movie_info.loc[mid]
            results.append({
                "movie_id": int(mid),
                "title":    str(row["title"]),
                "genres":   str(row["genres"]),
            })
        if len(results) >= top_n:
            break

    query_title = state.movie_info.loc[movie_id, "title"] if movie_id in state.movie_info.index else str(movie_id)
    return {
        "query":   {"movie_id": movie_id, "title": query_title},
        "similar": results,
    }
