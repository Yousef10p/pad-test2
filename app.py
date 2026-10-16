"""Streamlit front end for the FoundPAD document PAD: upload images, see each one with its score.

    streamlit run app.py

Tabs: one per model - exp6_3_4 and exp6_3_7 (best on every SaudiDocs test/ metric), both CLIP
ViT-L/14-336 - and comparison: upload the images once, pick a model for each of the two columns,
and every row shows the same image with each model's prediction. Every checkpoint is in the public
Hugging Face repo (a local copy is used when there is one).

Memory: one ViT-L is ~2 GB and Streamlit Community Cloud allows ~2.7 GB, so only one model is held
at a time. Scores are remembered per (checkpoint, image), so a rerun, a new threshold or the other
comparison column only loads a model for images it has not scored yet. Each tab's threshold is on
P(bona fide) - an image is called bona fide when the score is at or above it.
"""
import gc
import hashlib
import os
import threading

import streamlit as st
import torch

from foundpad_inference import FoundPADInference, THRESHOLD

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = 16                                     # images scored together, as in main.py
TYPES = ["jpg", "jpeg", "png", "bmp", "webp", "heic", "heif"]
MODELS = {                                     # model name -> checkpoint (a local .pth or a file in the HF repo)
    "exp6_3_4": "exp6_3_4/best.pth",           # balanced batches, ~70 % SaudiDocs (test/ AUC 0.9987)
    "exp6_3_7 (best in metrics)": "exp6_3_7/best.pth",   # exp6_3_4 + BID / RID (test/ AUC 0.9989, ACER 0.042)
}


@st.cache_resource
def model_slot():
    """The one loaded model, shared by every tab and session; the lock keeps a session from swapping
    the model while another one is scoring with it."""
    return {"key": None, "pad": None, "lock": threading.Lock()}


@st.cache_resource
def score_memory():
    """(checkpoint, mtime, image sha1) -> (P(bona fide), None) or (None, error message)."""
    return {}


def ckpt_key(checkpoint):
    """mtime is part of the key: a replaced local .pth is loaded (and scored) again."""
    return checkpoint, os.path.getmtime(checkpoint) if os.path.isfile(checkpoint) else None


def scores(checkpoint, files, label):
    """One (score, error) per uploaded file; the model is loaded only for images not scored yet."""
    memory, key = score_memory(), ckpt_key(checkpoint)
    ids = [(*key, hashlib.sha1(f.getvalue()).hexdigest()) for f in files]
    todo = [(i, f) for i, f in zip(ids, files) if i not in memory]
    if todo:
        slot = model_slot()
        with slot["lock"]:
            if slot["key"] != key:
                slot["pad"] = slot["key"] = None          # free the old model before loading the next
                gc.collect()
                with st.spinner(f"Loading {label} (the first run downloads CLIP)..."):
                    slot["pad"] = FoundPADInference(checkpoint, download_dir=os.path.join(HERE, "models"))
                slot["key"] = key
            pad = slot["pad"]
            ready = []                                    # preprocess first: an unreadable file is reported
            for i, f in todo:
                try:
                    ready.append((i, pad.preprocess(f.getvalue())))
                except (FileNotFoundError, ValueError) as e:
                    memory[i] = (None, str(e))
            progress = st.progress(0.0, text=f"Scoring with {label}: 0/{len(ready)} images")
            for b in range(0, len(ready), BATCH):
                chunk = ready[b:b + BATCH]
                for (i, _), r in zip(chunk, pad.infer(torch.cat([x for _, x in chunk]))):
                    memory[i] = (r["score_bona_fide"], None)
                done = min(b + BATCH, len(ready))
                progress.progress(done / len(ready), text=f"Scoring with {label}: {done}/{len(ready)} images")
            progress.empty()
    return [memory[i] for i in ids]


def card(f, score, error, threshold):
    """One image and its prediction."""
    st.image(f.getvalue(), width="stretch")
    if error:
        st.warning(error)
    elif score >= threshold:
        st.success(f"**bona fide** - P(bona fide) {100 * score:.4f}")
    else:
        st.error(f"**attack** - P(bona fide) {100 * score:.4f}")
    if not error:
        st.progress(score)
    st.caption(f.name)


def counts(results, threshold, cols):
    scored = [s for s, e in results if e is None]
    bona = sum(s >= threshold for s in scored)
    a, b, c = cols
    a.metric("Images scored", len(scored))
    b.metric("Bona fide", bona)
    c.metric("Attack", len(scored) - bona)


def threshold_slider(key):
    return st.slider("Threshold on P(bona fide)", 0.0, 1.0, THRESHOLD, 0.01, key=key,
                     help="Bona fide when the score is at or above this.")


def model_tab(name, default_checkpoint):
    """One model: settings, uploader, then every image with its score, three per row."""
    with st.expander("Settings"):
        checkpoint = st.text_input("Checkpoint", default_checkpoint, key=f"{name}_ckpt",
                                   help="A local .pth, or a file inside the Hugging Face repo.")
        threshold = threshold_slider(f"{name}_thr")
    files = st.file_uploader("Images", type=TYPES, accept_multiple_files=True, key=f"{name}_files")
    if not files:
        st.info("Waiting for images.")
        return
    try:
        results = scores(checkpoint, files, name)
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Could not load the model: {e}")
        return
    counts(results, threshold, st.columns(3))
    st.divider()
    for (f, (s, e)), col in zip(zip(files, results), [c for _ in range(0, len(files), 3) for c in st.columns(3)]):
        with col:
            card(f, s, e, threshold)


def comparison_tab():
    """Two columns, a model each; the images are uploaded once and every row is one image."""
    with st.expander("Settings"):
        threshold = threshold_slider("cmp_thr")
    left, right = st.columns(2)
    names = list(MODELS)
    picked = [left.selectbox("Left model", names, index=0, key="cmp_left"),
              right.selectbox("Right model", names, index=min(1, len(names) - 1), key="cmp_right")]
    files = st.file_uploader("Images", type=TYPES, accept_multiple_files=True, key="cmp_files")
    if not files:
        st.info("Waiting for images.")
        return
    try:
        results = [scores(MODELS[name], files, name) for name in picked]
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Could not load the model: {e}")
        return
    for col, name, res in zip(st.columns(2), picked, results):
        with col:
            st.subheader(name)
            counts(res, threshold, st.columns(3))
    st.divider()
    for k, f in enumerate(files):
        for col, res in zip(st.columns(2), results):
            with col:
                card(f, *res[k], threshold)
        st.divider()


st.set_page_config(page_title="Document PAD", page_icon="🪪", layout="wide")
st.title("🪪 Document presentation-attack detection")
st.caption("Pick a model, then upload one or more images of an ID document. Each is scored with "
           "FoundPAD + LoRA: P(bona fide) is high for a genuine capture, low for a screen / print attack.")

tabs = st.tabs([*MODELS, "comparison"])
for tab, (name, checkpoint) in zip(tabs, MODELS.items()):
    with tab:
        model_tab(name, checkpoint)
with tabs[-1]:
    comparison_tab()
