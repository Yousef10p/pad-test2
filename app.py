"""Streamlit front end for the FoundPAD document PAD: upload images, see each one with its score.

    streamlit run app.py

Two tabs, one model each: the current model (EXP5) and EXP6_3_2 (CLIP ViT-L/14-336), both
checkpoints in the public Hugging Face repo (a local copy is used when there is one). A tab loads its model only once images are uploaded there,
and each model is cached across reruns. Each tab's settings set the decision threshold on
P(bona fide) - an image is called bona fide when the score is at or above it.
"""
import os

import streamlit as st
import torch

from foundpad_inference import FoundPADInference, HF_FILE, THRESHOLD

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = 16                                     # images scored together, as in main.py
TYPES = ["jpg", "jpeg", "png", "bmp", "webp", "heic", "heif"]
MODELS = {                                     # tab -> default checkpoint (a local .pth or a file in the HF repo)
    "Current model (EXP5)": HF_FILE,
    "EXP6_3_2": "exp6_3_2/best.pth",           # the local copy if run from this folder, else from the HF repo
}


@st.cache_resource(max_entries=1, show_spinner="Loading the model (the first run downloads CLIP)...")
def load_model(checkpoint, mtime):
    """The FoundPADInference of one checkpoint, kept alive across reruns. Only one model is held at
    a time (ViT-L alone is ~2 GB; Streamlit Community Cloud allows ~2.7 GB), so switching tabs
    reloads. mtime (the local file's modification time) is only part of the cache key: a replaced
    .pth is loaded again."""
    return FoundPADInference(checkpoint, download_dir=os.path.join(HERE, "models"))


def pad_tab(key, default_checkpoint):
    """One tab: settings, uploader, then every image with its score."""
    with st.expander("Settings"):
        checkpoint = st.text_input("Checkpoint", default_checkpoint, key=f"{key}_ckpt",
                                   help="A local .pth, or a file inside the Hugging Face repo.")
        threshold = st.slider("Threshold on P(bona fide)", 0.0, 1.0, THRESHOLD, 0.01, key=f"{key}_thr",
                              help="Bona fide when the score is at or above this.")

    files = st.file_uploader("Images", type=TYPES, accept_multiple_files=True, key=f"{key}_files")
    if not files:
        st.info("Waiting for images.")
        return

    try:
        pad = load_model(checkpoint, os.path.getmtime(checkpoint) if os.path.isfile(checkpoint) else None)
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Could not load the model: {e}")
        return
    pad.threshold = float(threshold)           # the slider wins over the cached model's value
    st.caption(f"`{os.path.basename(os.path.dirname(pad.checkpoint))}/{os.path.basename(pad.checkpoint)}`"
               + (f" (epoch {pad.epoch})" if pad.epoch is not None else "")
               + f" - {pad.clip_id.split('/')[-1]} at {pad.input_size}px on **{pad.device}**")

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


st.set_page_config(page_title="Document PAD", page_icon="🪪", layout="wide")
st.title("🪪 Document presentation-attack detection")
st.caption("Pick a model, then upload one or more images of an ID document. Each is scored with "
           "FoundPAD + LoRA: P(bona fide) is high for a genuine capture, low for a screen / print attack.")

for tab, (name, checkpoint) in zip(st.tabs(list(MODELS)), MODELS.items()):
    with tab:
        pad_tab(name, checkpoint)
