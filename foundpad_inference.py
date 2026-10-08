"""FoundPAD + LoRA document PAD - inference, exactly as the EXP5 notebook
(TPO expirment/exp5_reproduceExp1/notebook2536ecccea.ipynb) scores its SaudiDocs test/ frames.

    pad = FoundPADInference()                              # exp5_reproduceExp1withMoreData_kaggle_oldSaudiDocs/best.pth
    pad = FoundPADInference("EXP1/best.pth")               # another checkpoint of the same pipeline
    pad = FoundPADInference("C:/models/my_checkpoint.pth")   # or any local file
    x = pad.preprocess("id_photo.jpg")                     # raises ValueError / FileNotFoundError
    print(pad.infer(x))    # [{'score_bona_fide': 0.93, 'decision': 'bona fide', 'threshold': 0.5}]

checkpoint: a local file, or a file inside HF_REPO. Looked for, in order: the path as given,
DOWNLOAD_DIR/<path> (downloaded before - no network), else downloaded from HF_REPO into
DOWNLOAD_DIR. Every FoundPAD checkpoint in the repo works - all come from the same pipeline.
A private repo needs a token: token="hf_...", or the HF_TOKEN environment variable.

The checkpoint is what the notebook saves: {"model": state_dict, ...}. Tiny checkpoints hold the
LoRA + head weights only (the frozen CLIP is rebuilt from CLIP_ID); a full state dict loads too.

Score = softmax P(bona fide), as in the paper: HIGH means bona fide, and an image is called
bona fide when score >= threshold. The default threshold is the fixed 0.5 - the per-set EER
thresholds in results.json were chosen WITH SaudiDocs labels and are not deployable.

Requires: torch torchvision "transformers>=4.44" "peft>=0.13" pillow huggingface_hub
          (pillow-heif for HEIC)
"""
import io
import os

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None              # as the notebook: phone photos can be 200 MP

# ---------- the notebook's configuration - must match the trained model ----------
CLIP_ID = "openai/clip-vit-base-patch16"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 8, 0.4
LORA_TARGETS = ["q_proj", "v_proj"]        # all 12 vision attention blocks
FRAME_SIZE = 256                           # frames cached at 256x256 (JPEG q95), per the paper
INPUT_SIZE = 224                           # model input
CLIP_MEAN = (0.4815, 0.4578, 0.4082)       # CLIP's native statistics, NOT ImageNet
CLIP_STD = (0.2686, 0.2613, 0.2758)
BONA = 0                                   # label 0 = bona fide; the score is softmax[:, BONA]
THRESHOLD = 0.5                            # fixed decision threshold

# ---------- where the trained checkpoints live ----------
HF_REPO = "US10F/TPO_to_PID2"
HF_FILE = "exp5_reproduceExp1withMoreData_kaggle_oldSaudiDocs/best.pth"   # EXP5 (Kaggle), epoch 1
DOWNLOAD_DIR = "models"                          # downloads land in DOWNLOAD_DIR/<path in the repo>


class FoundPADInference:
    def __init__(self, checkpoint=HF_FILE, device=None, threshold=THRESHOLD, repo=HF_REPO,
                 token=None, download_dir=DOWNLOAD_DIR):
        """Rebuild CLIP ViT-B/16 + rank-stabilized LoRA + linear head and load the checkpoint -
        a local file, or a file inside `repo`, downloaded into `download_dir` the first time."""
        import importlib.metadata
        import importlib.util
        import subprocess
        import sys

        # an old torchao (Colab ships 0.10) makes peft refuse to build LoRA layers; unused here
        if importlib.util.find_spec("torchao") is not None:
            from packaging.version import Version
            if Version(importlib.metadata.version("torchao")) < Version("0.16.0"):
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)
                for m in [m for m in sys.modules if m == "torchao" or m.startswith("torchao.")]:
                    del sys.modules[m]
                importlib.invalidate_caches()
        try:                                               # HEIC / HEIF phone photos, if available
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        from peft import LoraConfig, get_peft_model
        from transformers import CLIPModel

        checkpoint = str(checkpoint)
        downloaded = os.path.normpath(os.path.join(download_dir, checkpoint))
        if not os.path.isfile(checkpoint) and os.path.isfile(downloaded):
            checkpoint = downloaded                        # fetched on an earlier run
        elif not os.path.isfile(checkpoint):               # not on disk: fetch it from the repo
            from huggingface_hub import hf_hub_download
            try:
                checkpoint = hf_hub_download(repo, checkpoint.replace("\\", "/"), token=token,
                                             local_dir=download_dir)
            except Exception as e:
                raise FileNotFoundError(
                    f"checkpoint {checkpoint!r} is not a local file and could not be downloaded "
                    f"from {repo}: {type(e).__name__}: {e}. Check the name (it is the path inside "
                    f"the repo, e.g. {HF_FILE}); a private repo needs token= or HF_TOKEN.") from e
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(threshold)

        clip = CLIPModel.from_pretrained(CLIP_ID)
        for p in clip.parameters():
            p.requires_grad = False
        # ModuleDict names its children "clip." and "head." - the notebook's FoundPAD key names
        self.model = torch.nn.ModuleDict({
            "clip": get_peft_model(clip, LoraConfig(
                r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                target_modules=r".*vision_model.*\.(" + "|".join(LORA_TARGETS) + r")",
                use_rslora=True, bias="none")),
            "head": torch.nn.Linear(clip.config.projection_dim, 2)})

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        lost = [k for k in missing if "lora_" in k or k.startswith("head.")]
        if unexpected or lost:                  # only the frozen CLIP may be absent from the file
            raise ValueError(f"{checkpoint} does not match FoundPAD: {len(lost)} LoRA/head weights "
                             f"missing, {len(unexpected)} unexpected (e.g. {(lost + unexpected)[:2]})")
        self.model.to(self.device).eval()
        self.epoch = ck.get("epoch") if isinstance(ck, dict) else None
        self.to_tensor = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(CLIP_MEAN, CLIP_STD)])

    def preprocess(self, image):
        """One image -> a (1, 3, 224, 224) tensor, exactly as the notebook built its frames:
        upright by EXIF orientation -> RGB -> 256x256 bicubic -> JPEG q95 (the notebook's frame
        cache) -> 224x224 -> CLIP normalization. An image already 256x256 is used as is (no EXIF
        turn, no JPEG), as the notebook does.

        image: a file path, raw bytes, or a PIL image.
        Raises FileNotFoundError for a missing path, ValueError for anything that is not a
        readable image (empty, corrupt, truncated, unsupported format, absurdly large)."""
        try:
            if isinstance(image, Image.Image):
                img = image
            elif isinstance(image, (bytes, bytearray)):
                if not image:
                    raise ValueError("empty image data")
                img = Image.open(io.BytesIO(image))
            elif isinstance(image, (str, os.PathLike)):
                if not os.path.isfile(image):
                    raise FileNotFoundError(f"no such image file: {image}")
                if os.path.getsize(image) == 0:
                    raise ValueError(f"empty file: {image}")
                img = Image.open(image)
            else:
                raise ValueError(f"unsupported input type {type(image).__name__} - "
                                 "give a path, bytes or a PIL image")
            img.load()                                     # decode NOW: a cut-off file raises here
            if img.width < 1 or img.height < 1:
                raise ValueError(f"image has no pixels ({img.width}x{img.height})")
            if img.size != (FRAME_SIZE, FRAME_SIZE):       # the notebook's frame cache
                img = ImageOps.exif_transpose(img)         # turn phone photos upright (EXIF Orientation)
                buf = io.BytesIO()
                img.convert("RGB").resize((FRAME_SIZE, FRAME_SIZE), Image.BICUBIC).save(buf, "JPEG", quality=95)
                img = Image.open(io.BytesIO(buf.getvalue()))
            img = img.convert("RGB")
            return self.to_tensor(img).unsqueeze(0)
        except (FileNotFoundError, ValueError):
            raise
        except UnidentifiedImageError as e:
            raise ValueError(f"not an image, or a format Pillow cannot read "
                             f"(HEIC needs pillow-heif): {e}") from e
        except Image.DecompressionBombError as e:
            raise ValueError(f"image too large to decode safely: {e}") from e
        except (OSError, SyntaxError) as e:                # truncated / corrupt data
            raise ValueError(f"corrupt or truncated image: {e}") from e

    @torch.no_grad()
    def infer(self, x):
        """x: a tensor from preprocess(), or several stacked with torch.cat. Returns one dict per
        image: P(bona fide) - the paper's score, HIGH means bona fide - and the decision at the
        threshold. For frames of ONE video, average their scores first, as the notebook does."""
        if not torch.is_tensor(x) or x.ndim != 4 or tuple(x.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE):
            raise ValueError(f"expected a (N, 3, {INPUT_SIZE}, {INPUT_SIZE}) tensor from preprocess(), "
                             f"got {tuple(x.shape) if torch.is_tensor(x) else type(x).__name__}")
        e = self.model["clip"].get_image_features(pixel_values=x.to(self.device))   # float32, as trained
        if not torch.is_tensor(e):                   # transformers >= 5 returns a ModelOutput
            e = e.pooler_output
        logits = self.model["head"](torch.nn.functional.normalize(e, dim=-1))
        scores = torch.softmax(logits, -1)[:, BONA].cpu().numpy()
        return [{"score_bona_fide": float(s),
                 "decision": "bona fide" if s >= self.threshold else "attack",
                 "threshold": self.threshold} for s in np.atleast_1d(scores)]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FoundPAD document PAD: is each image bona fide?")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=HF_FILE,
                    help=f"a local .pth, or a file inside {HF_REPO} (default {HF_FILE})")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args()
    pad = FoundPADInference(a.model, threshold=a.threshold)
    for path in a.images:
        try:
            r = pad.infer(pad.preprocess(path))[0]
            print(f"{r['decision']:9s}  P(bona fide) {r['score_bona_fide']:.4f}  {path}")
        except (FileNotFoundError, ValueError) as err:
            print(f"ERROR      {err}")
