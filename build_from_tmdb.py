#!/usr/bin/env python3
import os, time, requests, pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_KEY = os.environ.get("TMDB_API_KEY")
if not TMDB_KEY:
    raise RuntimeError("TMDB_API_KEY not set in environment")

TMDB_BASE = "https://api.themoviedb.org/3"
OUT_FILE = "movie_sim_df.pkl"

def tmdb_get(path, params=None, retries=4):
    if params is None:
        params = {}
    params.update({"api_key": TMDB_KEY, "language": "en-US"})
    url = TMDB_BASE + path
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=12)
            if r.status_code == 200:
                return r.json()
            else:
                print(f"tmdb_get: status {r.status_code} for {url}")
        except Exception as e:
            print("tmdb_get exception:", e)
        time.sleep(1.5 * attempt)
    r.raise_for_status()

def fetch_popular(pages=8):
    movies = []
    for p in range(1, pages + 1):
        print("Fetching popular page", p)
        data = tmdb_get("/movie/popular", {"page": p})
        for m in data.get("results", []):
            movies.append({
                "tmdb_id": m.get("id"),
                "title": m.get("title") or "",
                "overview": m.get("overview") or "",
                "release_date": m.get("release_date") or "",
                "poster_path": m.get("poster_path") or "",
                "popularity": m.get("popularity") or 0
            })
    df = pd.DataFrame(movies).drop_duplicates("tmdb_id").reset_index(drop=True)
    print("Fetched movies:", len(df))
    return df

def fetch_genres(df):
    genre_map = {}
    for mid in tqdm(df["tmdb_id"].tolist(), desc="fetching details"):
        try:
            d = tmdb_get(f"/movie/{mid}")
            genres = [g.get("name","") for g in d.get("genres",[])]
            genre_map[mid] = " ".join([g for g in genres if g])
        except Exception as e:
            print(f"Warning: failed to fetch details for {mid}: {e}")
            genre_map[mid] = ""
    df["genres"] = df["tmdb_id"].map(lambda i: genre_map.get(i,""))
    return df

def build_text(df):
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna("").astype(str)
    df["text"] = (df["overview"] + " " + df["genres"] + " " + df["year"]).str.replace(r"\s+", " ", regex=True).str.strip()
    return df

def build_similarity(df):
    vect = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf = vect.fit_transform(df["text"].values)
    sim = cosine_similarity(tfidf)
    return sim, vect

def save_artifact(df, sim, vect):
    artifact = {
        "sim_matrix": sim,
        "tmdb_ids": df["tmdb_id"].tolist(),
        "id_to_title": dict(zip(df["tmdb_id"], df["title"])),
        "id_to_poster": dict(zip(df["tmdb_id"], df["poster_path"])),
        "metadata_df": df,
        "vectorizer": vect
    }
    with open(OUT_FILE, "wb") as f:
        pickle.dump(artifact, f)
    print("Saved artifact to", OUT_FILE)

def main():
    pages = int(os.environ.get("TMDB_PAGES", "8"))
    df = fetch_popular(pages=pages)
    df = fetch_genres(df)
    df = build_text(df)
    sim, vect = build_similarity(df)
    save_artifact(df, sim, vect)

if __name__ == "__main__":
    main()
