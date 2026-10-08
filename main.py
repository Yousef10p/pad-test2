"""Score every image in a list of folders with FoundPAD, then report the metrics per folder and
over all folders together.

    python main.py        (or Run in VS Code)

The true answer is read from the file name, else from the folder name:
    bona / genuine / real            -> bona fide
    screen / print / replay / attack -> attack
An image whose name says neither is scored but left out of the metrics.
Attack is the positive class (ISO/IEC 30107-3): APCER = attacks accepted as bona fide,
BPCER = bona fide rejected, recall / precision are for catching attacks.
"""
import os

import numpy as np
import torch

from foundpad_inference import FoundPADInference

HERE = os.path.dirname(os.path.abspath(__file__))       # this folder, wherever it is run from
# FOLDERS = ["dataset1", "dataset2", "dataset3", "dataset4",   # folders next to this file,
#            "dataset5", "dataset6", "dataset7"]               # or full paths

FOLDERS = ["dataset_test_new"]
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif")
BATCH = 16                                              # images scored together
LIST_BELOW = 50              # a folder with fewer images than this lists every image and its score


def true_label(path):
    """1 = attack, 0 = bona fide, None = neither the file nor the folder name says."""
    for name in (os.path.basename(path), os.path.basename(os.path.dirname(path))):
        name = name.lower()
        if any(w in name for w in ("bona", "genuine", "real")):
            return 0
        if any(w in name for w in ("screen", "print", "replay", "attack")):
            return 1
    return None


def metrics(y, s, t):
    """y: 1 = attack, 0 = bona fide.  s: P(bona fide) - called attack when s < t.
    AUC and EER need both classes, APCER needs attacks, BPCER needs bona fide: None otherwise."""
    y, s = np.asarray(y), np.asarray(s)
    attack, bona, called_attack = y == 1, y == 0, s < t
    tp, fn = int((called_attack & attack).sum()), int((~called_attack & attack).sum())
    fp, tn = int((called_attack & bona).sum()), int((~called_attack & bona).sum())
    ratio = lambda a, b: a / b if b else None
    out = {"images": len(y), "wrong": f"{fn + fp}/{len(y)}", "accuracy": ratio(tp + tn, len(y)),
           "APCER": ratio(fn, tp + fn), "BPCER": ratio(fp, fp + tn), "EER": None, "AUC": None,
           "recall": ratio(tp, tp + fn), "precision": ratio(tp, tp + fp)}
    if attack.any() and bona.any():
        # the ROC over P(attack) = 1 - s, one point per distinct score: fpr = BPCER, 1 - tpr = APCER
        order = np.argsort(s, kind="mergesort")              # P(attack) high -> low
        ys, ss = y[order], s[order]
        cut = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
        tpr = np.r_[0, np.cumsum(ys)[cut] / attack.sum()]
        fpr = np.r_[0, np.cumsum(1 - ys)[cut] / bona.sum()]
        out["AUC"] = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
        i = int(np.argmin(np.abs(fpr - (1 - tpr))))
        out["EER"] = float((fpr[i] + 1 - tpr[i]) / 2)
    return out


def table(rows):
    cols = ["images", "wrong", "accuracy", "APCER", "BPCER", "EER", "AUC", "recall", "precision"]
    show = lambda v: "n/a" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)
    width = max(len(name) for name in rows) + 2
    print(f"{'':{width}s}" + "".join(f"{c:>12s}" for c in cols))
    for name, m in rows.items():
        print(f"{name:{width}s}" + "".join(f"{show(m[c]):>12s}" for c in cols))


if __name__ == "__main__":
    pad = FoundPADInference(download_dir=os.path.join(HERE, "models"))   # downloaded once, then reused
    print(f"model {pad.checkpoint} (epoch {pad.epoch}) on {pad.device}, threshold {pad.threshold}\n")

    RESULTS = {}                                         # folder -> (true labels, P(bona fide))
    for folder in FOLDERS:
        path = folder if os.path.isabs(folder) else os.path.join(HERE, folder)
        if not os.path.isdir(path):
            print(f"{folder}: not a folder - skipped")
            continue
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(IMAGE_EXT))
        y, s, unread, unlabelled, scored = [], [], 0, 0, []
        for b in range(0, len(files), BATCH):
            xs, labels, names = [], [], []
            for f in files[b:b + BATCH]:
                try:
                    xs.append(pad.preprocess(f))
                    labels.append(true_label(f))
                    names.append(os.path.basename(f))
                except (FileNotFoundError, ValueError) as e:
                    unread += 1
                    print(f"  ERROR {os.path.basename(f)}: {e}")
            for lab, name, r in zip(labels, names, pad.infer(torch.cat(xs)) if xs else []):
                scored.append((name, lab, r))
                if lab is None:
                    unlabelled += 1
                else:
                    y.append(lab)
                    s.append(r["score_bona_fide"])
            print(f"\r{folder}: {min(b + BATCH, len(files)):,}/{len(files):,} images", end="")
        print(f"\r{folder}: {len(files):,} images - {len(y):,} scored with a known answer"
              + (f", {unlabelled} with no bona/attack word in the name" if unlabelled else "")
              + (f", {unread} unreadable" if unread else ""))
        if len(files) < LIST_BELOW:                      # a small folder: every image, score first
            print("  P(bona fide)  decision   image")
            for name, lab, r in scored:
                truth = {0: "bona fide", 1: "attack", None: None}[lab]
                mark = "" if truth is None else ("" if r["decision"] == truth else f"   <- WRONG, is {truth}")
                print(f"  {r['score_bona_fide']:12.4f}  {r['decision']:9s}  {name}{mark}")
        if y:
            RESULTS[os.path.basename(os.path.normpath(path))] = (y, s)     # a short name for the table

    if RESULTS:
        rows = {name: metrics(y, s, pad.threshold) for name, (y, s) in RESULTS.items()}
        rows["ALL FOLDERS"] = metrics([v for y, _ in RESULTS.values() for v in y],
                                      [v for _, s in RESULTS.values() for v in s], pad.threshold)
        print(f"\nthreshold {pad.threshold} on P(bona fide) - attack is the positive class; "
              "n/a = needs both classes (AUC, EER) or the missing class (APCER, BPCER)\n")
        table(rows)
