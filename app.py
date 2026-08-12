import streamlit as st
import json
import os
import glob
import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

st.set_page_config(page_title="Swami's Katha", page_icon=None, layout="centered")

# ---------- Background image (put your own file at assets/background.jpg) ----------
BACKGROUND_CSS = """
<style>
.stApp {
    background-image: url("app/static/background.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(255, 255, 255, 0.85);
    z-index: -1;
}
h1, h2, h3 { color: #b8860b; text-align: center; }
</style>
"""
st.markdown(BACKGROUND_CSS, unsafe_allow_html=True)

st.markdown("<h1>Swami's Katha</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:14px;'>Jay Swaminarayan</p>", unsafe_allow_html=True)

# ---------- Load data (cached so it only loads once) ----------
@st.cache_resource
def load_chunks():
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_embed_model():
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

@st.cache_resource
def get_llm_client():
    api_key = st.secrets["NVIDIA_API_KEY"]
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)

chunks = load_chunks()
embed_model = load_embed_model()
llm = get_llm_client()

chunk_embeddings = np.array([c["embedding"] for c in chunks])

KATHA_SOURCES = sorted(list({c["date"] for c in chunks}))
DATE_TO_AUDIO = {c["date"]: c["audio_file"] for c in chunks}


def format_timestamp(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def cosine_search(query, top_n_dates=10):
    q_emb = embed_model.encode([query])[0]
    sims = chunk_embeddings @ q_emb / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
    )
    best_per_date = {}
    for i, c in enumerate(chunks):
        d = c["date"]
        score = sims[i]
        if d not in best_per_date or score > best_per_date[d][1]:
            best_per_date[d] = (c, score)
    ordered = sorted(best_per_date.values(), key=lambda x: x[0]["date"])
    return [c for c, s in ordered[:top_n_dates]]


def build_prompt(query, matches):
    context_blocks = []
    for c in matches:
        ts = format_timestamp(c["start_time"])
        block = f"[Katha date: {c['date']}, timestamp: {ts}]\n{c['text']}"
        context_blocks.append(block)
    context = "\n\n---\n\n".join(context_blocks)
    lines = [
        "Neeche Swamiji ki alag-alag kathaon ke exact transcript paragraphs diye gaye hain, jo user ke sawaal se related hain.",
        "",
        f"User ka sawaal: {query}",
        "",
        "Transcripts:",
        context,
        "",
        "Har katha ke liye ek chhota (1-2 line) summary do ki usme kya kaha gaya, phir end me ek overall (3-4 line) combined summary do.",
    ]
    return "\n".join(lines)


def build_combined_audio(matches, out_name):
    clip_paths = []
    for i, c in enumerate(matches):
        audio_path = f"data/audio/{c['audio_file']}"
        if not os.path.exists(audio_path):
            continue
        start = c["start_time"]
        end = c["end_time"]
        clip_path = f"/tmp/clip_{i}.mp3"
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-t", str(end - start),
            "-acodec", "libmp3lame", clip_path
        ], capture_output=True)
        clip_paths.append(clip_path)

    if not clip_paths:
        return None

    list_path = "/tmp/concat_list.txt"
    with open(list_path, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")

    combined_path = f"/tmp/{out_name}.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
        "-acodec", "libmp3lame", combined_path
    ], capture_output=True)
    return combined_path


tab1, tab2 = st.tabs(["Katha Library", "Ask Anything from Katha"])

with tab1:
    st.markdown("<h2>Katha Library</h2>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, date in enumerate(KATHA_SOURCES):
        with cols[i % 3]:
            st.markdown(f"**{date}**")
            audio_path = f"data/audio/{DATE_TO_AUDIO[date]}"
            if os.path.exists(audio_path):
                st.audio(audio_path)

with tab2:
    st.markdown("<h2>Ask Anything from Katha</h2>", unsafe_allow_html=True)
    query = st.text_input("Apna sawaal yahan likho... (e.g. krodh, vairagya, nav prakar ni bhakti)")
    if st.button("Search"):
        if not query.strip():
            st.warning("Pehle koi sawaal likho.")
        else:
            with st.spinner("Saari 10 kathaon me dhundh raha hoon..."):
                matches = cosine_search(query)

            if not matches:
                st.info("Koi matching content nahi mila.")
            else:
                with st.spinner("Audio clips jod raha hoon..."):
                    safe_name = "".join(c if c.isalnum() else "_" for c in query)[:30]
                    combined_path = build_combined_audio(matches, safe_name)

                with st.spinner("Jawaab taiyaar kar raha hoon..."):
                    prompt = build_prompt(query, matches)
                    response = llm.chat.completions.create(
                        model="meta/llama-3.3-70b-instruct",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1800,
                    )

                st.markdown(f"### '{query}' — {len(matches)} kathaon me mila")

                if combined_path:
                    st.markdown("**Sab moments ek saath sunein:**")
                    st.audio(combined_path)

                st.markdown("**Kaunse kathaon me mila:**")
                for c in matches:
                    ts_start = format_timestamp(c["start_time"])
                    ts_end = format_timestamp(c["end_time"])
                    st.markdown(f"- {c['date']} — {ts_start} to {ts_end}")

                st.markdown("---")
                st.write(response.choices[0].message.content)
