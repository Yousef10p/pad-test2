"""Streamlit front end for the FoundPAD document PAD: upload images, see each one with its score.

    streamlit run app.py

The model is loaded once per session and cached; the sidebar sets the decision threshold on
P(bona fide) - an image is called bona fide when the score is at or above it.
"""
import os

import streamlit as st
import torch

from foundpad_inference import FoundPADInference, HF_FILE, THRESHOLD

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = 16                                     # images scored together, as in main.py
TYPES = ["jpg", "jpeg", "png", "bmp", "webp", "heic", "heif"]


@st.cache_resource(show_spinner="Loading the model (the first run downloads CLIP)...")
def load_model(checkpoint):
    """One FoundPADInference per checkpoint, kept alive across reruns."""
    return FoundPADInference(checkpoint, download_dir=os.path.join(HERE, "models"))


st.set_page_config(page_title="Document PAD", page_icon="🪪", layout="wide")
st.title("🪪 Document presentation-attack detection")
st.caption("Upload one or more images of an ID document. Each is scored with FoundPAD + LoRA: "
           "P(bona fide) is high for a genuine capture, low for a screen / print attack.")

with st.sidebar:
    st.header("Model")
    checkpoint = st.text_input("Checkpoint", HF_FILE,
                               help="A local .pth, or a file inside the Hugging Face repo.")
    threshold = st.slider("Threshold on P(bona fide)", 0.0, 1.0, THRESHOLD, 0.01,
                          help="Bona fide when the score is at or above this.")

try:
    pad = load_model(checkpoint)
except (FileNotFoundError, ValueError) as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

pad.threshold = float(threshold)               # the slider wins over the cached model's value
with st.sidebar:
    st.success(f"`{os.path.basename(pad.checkpoint)}`"
               + (f" (epoch {pad.epoch})" if pad.epoch is not None else "")
               + f" on **{pad.device}**")

files = st.file_uploader("Images", type=TYPES, accept_multiple_files=True)
if not files:
    st.info("Waiting for images.")
    st.stop()

# preprocess first, so an unreadable file is reported instead of breaking the batch
ready, failed = [], []
for f in files:
    try:
        ready.append((f, pad.preprocess(f.getvalue())))
    except (FileNotFoundError, ValueError) as e:
        failed.append((f.name, str(e)))

results = []
progress = st.progress(0.0, text=f"Scoring 0/{len(ready)} images")
for b in range(0, len(ready), BATCH):
    chunk = ready[b:b + BATCH]
    for (f, _), r in zip(chunk, pad.infer(torch.cat([x for _, x in chunk]))):
        results.append((f, r))
    progress.progress(len(results) / len(ready), text=f"Scoring {len(results)}/{len(ready)} images")
progress.empty()

bona = sum(r["decision"] == "bona fide" for _, r in results)
a, b, c = st.columns(3)
a.metric("Images scored", len(results))
b.metric("Bona fide", bona)
c.metric("Attack", len(results) - bona)
st.divider()

for (f, r), col in zip(results, [c for row in range(0, len(results), 3)
                                 for c in st.columns(3)]):
    with col:
        st.image(f.getvalue(), width="stretch")
        score = r["score_bona_fide"]
        if r["decision"] == "bona fide":
            st.success(f"**bona fide** - P(bona fide) {100 * score:.4f}")
        else:
            st.error(f"**attack** - P(bona fide) {100 * score:.4f}")
        st.progress(score)
        st.caption(f.name)

for name, err in failed:
    st.warning(f"{name}: {err}")
