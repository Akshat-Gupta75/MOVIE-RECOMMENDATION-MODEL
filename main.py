import asyncio
import os
import pickle
import time
from functools import lru_cache
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

WATCHMODE_API_KEY = os.getenv("WATCHMODE_API_KEY")

WATCHMODE_BASE = "https://api.watchmode.com/v1"

if not WATCHMODE_API_KEY:
    raise RuntimeError("WATCHMODE_API_KEY missing. Add it to .env as WATCHMODE_API_KEY=your_key")

print(f"WATCHMODE KEY LOADED: {bool(WATCHMODE_API_KEY)}")
print(f"WATCHMODE KEY LENGTH: {len(WATCHMODE_API_KEY)}")

app = FastAPI(title="Movie Recommender API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pickle paths ---
DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")


_client: Optional[httpx.AsyncClient] = None

DETAILS_TTL = 3600
HOME_CACHE_TTL = 300
_details_cache: Dict[int, tuple] = {}
_home_cache: Dict[tuple, tuple] = {}


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=20)
    return _client


def _ttl_get(cache: Dict, key) -> Any:
    item = cache.get(key)
    if not item:
        return None
    value, expires = item
    if time.monotonic() > expires:
        cache.pop(key, None)
        return None
    return value


def _ttl_set(cache: Dict, key, value, ttl: float) -> None:
    cache[key] = (value, time.monotonic() + ttl)


async def _run_capped(coros, limit: int = 8):
    sem = asyncio.Semaphore(limit)

    async def _wrapped(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*(_wrapped(c) for c in coros))


df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None



# --- Pydantic models ---
class WATCHMODEmoviecard(BaseModel):
    watchmode_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class WATCHMODEmoviedeatils(BaseModel):
    watchmode_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    watchmode: Optional[WATCHMODEmoviecard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: WATCHMODEmoviedeatils
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[WATCHMODEmoviecard]



# --- Utility ---
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(url):
    if not url:
        return None
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        return url
    return None


def _resolve_poster(raw: Dict[str, Any]) -> str:
    return make_img_url(
        raw.get("poster")
        or raw.get("posterMedium")
        or raw.get("posterLarge")
        or raw.get("poster_large")
        or raw.get("image_url")
    )


async def enrich_cards_with_posters(
    cards: List[WATCHMODEmoviecard],
    concurrency: int = 16,
) -> List[WATCHMODEmoviecard]:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(card: WATCHMODEmoviecard) -> WATCHMODEmoviecard:
        if card.poster_url:
            return card
        async with sem:
            for attempt in (0, 1):  # one retry on transient failure
                try:
                    data = await _get_details_payload(card.watchmode_id)
                    poster = _resolve_poster(data)
                    if poster:
                        card.poster_url = poster
                    break
                except Exception:
                    if attempt == 1:
                        break
        return card

    return await asyncio.gather(*(guarded(c) for c in cards))


async def watchmode_get(path: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    q = dict(params or {})
    try:
        client = _get_client()
        r = await client.get(
            f"{WATCHMODE_BASE}{path}",
            params=q,
            headers={"X-API-Key": WATCHMODE_API_KEY},
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=501,
            detail=f"WATCHMODE request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"WATCHMODE error: {r.status_code}: {r.text[:500]}",
        )
    try:
        return r.json()
    except Exception:
        raise HTTPException(
            status_code=502,
            detail=f"WATCHMODE returned non-JSON response: {r.status_code}",
        )


async def watchmode_cards_from_results(
    results: List[dict] = [], limit: int = 20
) -> List[WATCHMODEmoviecard]:
    out: List[WATCHMODEmoviecard] = []
    for m in (results or [])[:limit]:
        wm_id = m.get("id")
        if not wm_id:
            continue
        out.append(
            WATCHMODEmoviecard(
                watchmode_id=int(wm_id),
                title=m.get("title") or m.get("name") or "",
                poster_url=_resolve_poster(m),
                release_date=m.get("release_date"),
                vote_average=m.get("user_rating"),
            )
        )
    return out


def _genres_from_details(data: Dict[str, Any]) -> List[dict]:
    genre_ids = data.get("genres") or []
    genre_names = data.get("genre_names") or []
    out: List[dict] = []
    if genre_names and genre_ids and len(genre_names) == len(genre_ids):
        for gid, gname in zip(genre_ids, genre_names):
            out.append({"id": int(gid), "name": str(gname)})
    elif genre_names:
        for gname in genre_names:
            out.append({"id": None, "name": str(gname)})
    elif genre_ids:
        for gid in genre_ids:
            out.append({"id": int(gid), "name": None})
    return out


def _clean_overview(data: Dict[str, Any]) -> str:
    """Return a clean overview string from a Watchmode details dict."""
    raw = (
        data.get("plot_overview")
        or data.get("plot")
        or data.get("overview")
        or data.get("description")
    )
    if not isinstance(raw, str):
        return ""
    return raw.strip()


async def _get_details_payload(watchmode_id: int) -> Dict[str, Any]:
    cached = _ttl_get(_details_cache, watchmode_id)
    if cached is not None:
        return cached
    data = await watchmode_get(f"/title/{watchmode_id}/details", {"language": "en"})
    _ttl_set(_details_cache, watchmode_id, data, DETAILS_TTL)
    return data


async def watchmode_get_movie_details(watchmode_id: int) -> WATCHMODEmoviedeatils:
    data = await _get_details_payload(watchmode_id)
    return WATCHMODEmoviedeatils(
        watchmode_id=int(data["id"]),
        title=data.get("title") or data.get("name") or "",
        overview=_clean_overview(data),
        release_date=data.get("release_date"),
        poster_url=_resolve_poster(data),
        backdrop_url=make_img_url(
            data.get("backdrop") or data.get("backdrop_image_url")
        ),
        genres=_genres_from_details(data),
    )


async def watchmode_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    data = await watchmode_get(
        "/search",
        {
            "search_field": "name",
            "search_value": query,
            "types": "movie",
        },
    )
    titles = data.get("title_results") or []
    return {"results": titles, "page": page, "total_pages": 1, "total_results": len(titles)}


async def watchmode_search_first(query: str) -> Optional[dict]:
    data = await watchmode_search_movies(query=query)
    results = data.get("results") or []
    return results[0] if results else None


# --- Title-to-index map ---
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:

    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    
    except Exception:
        raise RuntimeError("indices.pkl must be dict or pandas Series-like (with .items())")


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


@lru_cache(maxsize=256)
def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")
    idx = get_local_idx_by_title(query_title)
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)
    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_watchmode_card_by_title(title: str) -> Optional[WATCHMODEmoviecard]:
    try:
        m = await watchmode_search_first(title)
        if not m:
            return None
        return WATCHMODEmoviecard(
            watchmode_id=int(m["id"]),
            title=m.get("name") or m.get("title") or title,
            poster_url=_resolve_poster(m),
            release_date=m.get("release_date"),
            vote_average=m.get("user_rating"),
        )
    except Exception:
        return None


# --- Startup: Load pickles ---
@app.on_event("startup")
def load_pickle():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")


@app.on_event("shutdown")
async def close_http_client():
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:
            pass
        _client = None

# ==================== ROUTES ====================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/home", response_model=List[WATCHMODEmoviecard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Home feed for Streamlit (posters).
    Uses the Watchmode /list-titles endpoint (types=movie).
    Note: The Watchmode list-titles endpoint does not return poster
    URLs, so poster_url will be null unless fetched via details.
    """
    try:
        valid = {"popular", "trending", "top_rated", "upcoming", "now_playing"}
        if category not in valid:
            raise HTTPException(status_code=400, detail="Invalid category")

        cache_key = (category, limit)
        cached = _ttl_get(_home_cache, cache_key)
        if cached is not None:
            cards = cached
        else:
            data = await watchmode_get(
                "/list-titles",
                {"types": "movie", "limit": min(limit, 250), "page": 1},
            )
            cards = await watchmode_cards_from_results(
                data.get("titles", []), limit=limit
            )
            _ttl_set(_home_cache, cache_key, cards, HOME_CACHE_TTL)

        # Self-heal: re-attempt posters for any card still missing one.
        # Genuine no-poster titles hit the details cache (no network).
        await enrich_cards_with_posters(cards)
        return cards

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# @app.get("/home", response_model=List[WATCHMODEmoviecard])
# async def home(
#     category: str = Query("popular"),
#     limit: int = Query(12, ge=1, le=50),
# ):
#     try:
#         category_keywords = {
#             "popular": ["action movies", "adventure movies", "comedy movies", "drama movies"],
#             "trending": ["thriller movies", "sci-fi movies", "fantasy movies", "mystery movies"],
#             "latest": ["new movies 2025", "new movies 2024", "upcoming movies", "recent films"],
#             "alphabetical": ["classic movies", "award winning movies", "best movies", "top rated"],
#         }

#         if category not in category_keywords:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Invalid category. Use: popular, trending, latest, alphabetical",
#             )

#         keywords = category_keywords[category]
#         random.shuffle(keywords)
#         search_keywords = keywords[:3]

#         seen_ids = set()
#         cards: List[WATCHMODEmoviecard] = []

#         for kw in search_keywords:
#             if len(cards) >= limit:
#                 break
#             try:
#                 data = await watchmode_get(
#                     "/v1/search/",
#                     params={"s": kw},
#                 )
#                 results = data.get("results") or []
#                 for m in results:
#                     if len(cards) >= limit:
#                         break
#                     wm_id = m.get("id")
#                     if not wm_id or wm_id in seen_ids:
#                         continue
#                     seen_ids.add(wm_id)
#                     poster = make_img_url(m.get("image_url") or m.get("poster") or m.get("poster_large"))
#                     cards.append(
#                         WATCHMODEmoviecard(
#                             watchmode_id=int(wm_id),
#                             title=m.get("title") or "Unknown",
#                             poster_url=poster,
#                             release_date=m.get("release_date") or "",
#                             vote_average=m.get("user_rating") or 0,
#                         )
#                     )
#             except HTTPException:
#                 continue

#         return cards[:limit]

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Home route failed: {str(e)}")

@app.get("/watchmode/search") #response_model=List[WATCHMODEmoviecard]#)
async def watchmode_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    data = await watchmode_get(
        "/autocomplete-search",
        {
            "search_value": query,
            "search_type": 3,  # titles only, movies
        },
    )
    return [
        WATCHMODEmoviecard(
            watchmode_id=int(m["id"]),
            title=(m.get("name") or m.get("title") or "").strip(),
            poster_url=_resolve_poster(m),
            release_date=str(m["year"]) if m.get("year") else None,
            vote_average=m.get("user_rating"),
        )
        for m in (data.get("results") or [])[:20]
        if m.get("id")
    ]
    

@app.get("/movie/id/{watchmode_id}", response_model=WATCHMODEmoviedeatils)
async def movie_details_route(watchmode_id: int):
    return await watchmode_get_movie_details(watchmode_id)


@app.get("/recommend/genre", response_model=List[WATCHMODEmoviecard])
async def recommend_genre(
    watchmode_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    details = await watchmode_get_movie_details(watchmode_id)
    if not details.genres:
        return []

    genre_id = details.genres[0].get("id")
    if not genre_id:
        return []

    discover = await watchmode_get(
        "/list-titles",
        {
            "types": "movie",
            "genres": genre_id,
            "limit": min(limit, 250),
            "page": 1,
        },
    )
    cards = await watchmode_cards_from_results(discover.get("titles", []), limit=limit)
    filtered = [c for c in cards if c.watchmode_id != watchmode_id]
    return await enrich_cards_with_posters(filtered)


@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    best = await watchmode_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No watchmode movie found for query: {query}"
        )

    watchmode_id = int(best["id"])
    details = await watchmode_get_movie_details(watchmode_id)

    # TF-IDF recommendations
    tfidf_items: List[TFIDFRecItem] = []
    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    try:
        cards_attached = await _run_capped(
            [attach_watchmode_card_by_title(t) for t, _ in recs], limit=8
        )
    except Exception:
        cards_attached = [None] * len(recs)
    tfidf_items = [
        TFIDFRecItem(title=t, score=s, watchmode=c)
        for (t, s), c in zip(recs, cards_attached)
    ]

    # Genre recommendations via Watchmode list-titles with genre filter
    genre_recs: List[WATCHMODEmoviecard] = []
    if details.genres:
        genre_id = details.genres[0].get("id")
        if genre_id:
            discover = await watchmode_get(
                "/list-titles",
                {
                    "types": "movie",
                    "genres": genre_id,
                    "limit": min(genre_limit, 250),
                    "page": 1,
                },
            )
            cards = await watchmode_cards_from_results(
                discover.get("titles", []), limit=genre_limit
            )
            genre_recs = [c for c in cards if c.watchmode_id != details.watchmode_id]

    cards_to_enrich: List[WATCHMODEmoviecard] = [
        item.watchmode for item in tfidf_items if item.watchmode is not None
    ]
    cards_to_enrich.extend(genre_recs)
    seen_ids = set()
    unique_cards: List[WATCHMODEmoviecard] = []
    for card in cards_to_enrich:
        if card.watchmode_id in seen_ids:
            continue
        seen_ids.add(card.watchmode_id)
        unique_cards.append(card)
    if unique_cards:
        await enrich_cards_with_posters(unique_cards)

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
