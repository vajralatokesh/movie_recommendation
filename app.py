#!/usr/bin/env python3
import os, pickle, numpy as np, gradio as gr

OUT_FILE = "movie_sim_df.pkl"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

if not os.path.exists(OUT_FILE):
    raise RuntimeError(f"{OUT_FILE} not found. Run build_from_tmdb.py first and place the file in the repo root.")

with open(OUT_FILE, "rb") as f:
    data = pickle.load(f)

sim = data["sim_matrix"]
tmdb_ids = data["tmdb_ids"]
id_to_title = data["id_to_title"]
id_to_poster = data["id_to_poster"]
tmdb_to_index = {tmdb_id: idx for idx, tmdb_id in enumerate(tmdb_ids)}
title_list = sorted(list(id_to_title.values()))

def find_match(input_title):
    # exact then substring
    for tid, t in id_to_title.items():
        if input_title.lower() == t.lower():
            return tid
    for tid, t in id_to_title.items():
        if input_title.lower() in t.lower():
            return tid
    return None

def recommend(title_query, n=8):
    if not title_query:
        return [], "Please enter a movie title."
    tid = find_match(title_query)
    if tid is None:
        return [], f"No match for '{title_query}'. Try selecting from dropdown."
    idx = tmdb_to_index[tid]
    scores = sim[idx]
    top_idxs = np.argsort(-scores)
    top_idxs = [i for i in top_idxs if i != idx][:n]
    results = []
    for i in top_idxs:
        mid = tmdb_ids[i]
        title = id_to_title.get(mid, "Unknown")
        poster = id_to_poster.get(mid)
        poster_url = IMAGE_BASE + poster if poster else None
        results.append((poster_url, title))
    return results, f"Recommendations for '{id_to_title[tid]}'"

with gr.Blocks() as demo:
    gr.Markdown("# Movie Recommender (TMDB)")
    with gr.Row():
        with gr.Column(scale=1):
            dropdown = gr.Dropdown(choices=title_list, label="Pick a movie (or type)", searchable=True)
            text_in = gr.Textbox(placeholder="Or type a movie title (partial allowed)", label="Or type title")
            num = gr.Slider(minimum=1, maximum=20, value=8, step=1, label="Number of recommendations")
            btn = gr.Button("Recommend")
            status = gr.Label(value="Ready")
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Recommendations (poster + title)").style(grid=[4], height="auto")
    def on_click(drop_val, typed, n):
        q = typed.strip() if typed and typed.strip() else drop_val
        results, msg = recommend(q, int(n))
        gallery_items = [(img, cap) for img, cap in results]
        return gallery_items, msg
    btn.click(fn=on_click, inputs=[dropdown, text_in, num], outputs=[gallery, status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
