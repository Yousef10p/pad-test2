"""FoundPAD + LoRA document PAD - inference, exactly as the training notebooks score their
SaudiDocs test/ frames (the default model: exp6_3_7/EXP6_3_7.ipynb).

    pad = FoundPADInference()                              # exp6_3_7/best.pth (EXP6_3_7)
    pad = FoundPADInference("exp6_3_4/best.pth")           # another checkpoint of the same pipeline
    pad = FoundPADInference("C:/models/my_checkpoint.pth")   # or any local file
    x = pad.preprocess("id_photo.jpg")                     # raises ValueError / FileNotFoundError
    print(pad.infer(x))    # [{'score_bona_fide': 0.93, 'decision': 'bona fide', 'threshold': 0.5}]

checkpoint: a local file, or a file inside HF_REPO. Looked for, in order: the path as given,
DOWNLOAD_DIR/<path> (downloaded before - no network), else downloaded from HF_REPO into
DOWNLOAD_DIR. Every FoundPAD checkpoint in the repo works - all come from the same pipeline.
A private repo needs a token: token="hf_...", or the HF_TOKEN environment variable. So does
DINOv3, which is gated: the token's account must have accepted its license on Hugging Face.

The checkpoint is what the notebook saves: {"model": state_dict, ...}. Tiny checkpoints hold the
LoRA + head weights only (the frozen backbone is rebuilt from its HF model id); a full state dict
loads too. The backbone, its normalization and its image sizes come from the checkpoint itself
(its LoRA width -> BACKBONES), so one class serves every kind:
    768  = CLIP ViT-B/16       frame 256 -> input 224   (EXP1, EXP5)
    1024 = CLIP ViT-L/14-336   frame 384 -> input 336   (EXP6_3_2 ... EXP6_3_7)
    1280 = DINOv3 ViT-H+/16    frame 384 -> input 336   (EXP6_3_7_DINOV3_VITH_PLUS_KAGGLE)
EXP6_3_7 = EXP6_3_4 + BID and RID in training; its preprocessing and scoring are EXP6_3_4's.
The DINOv3 run is EXP6_3_7 with the backbone swapped: the head sits on DINOv3's pooler_output (the
CLS token after the final norm, 1280-D) instead of CLIP's projected image embedding, and the
normalization is DINOv3's (ImageNet) instead of CLIP's.

Score = softmax P(bona fide), as in the paper: HIGH means bona fide, and an image is called
bona fide when score >= threshold. The default threshold is the fixed 0.5 - the per-set EER
thresholds in results.json were chosen WITH SaudiDocs labels and are not deployable.

Requires: torch torchvision "transformers>=4.56" "peft>=0.13" pillow huggingface_hub
          (pillow-heif for HEIC; transformers 4.56 is the first with DINOv3)
"""
import io
import os

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None              # as the notebook: phone photos can be 200 MP

CLIP_MEAN = (0.4815, 0.4578, 0.4082)       # CLIP's native statistics, NOT ImageNet
CLIP_STD = (0.2686, 0.2613, 0.2758)
DINO_MEAN = (0.485, 0.456, 0.406)          # DINOv3's own normalization (ImageNet)
DINO_STD = (0.229, 0.224, 0.225)

# ---------- the backbone - picked per checkpoint, never set by hand ----------
# backbone width (= a LoRA A matrix's input size, read from the checkpoint) ->
#   (HF model, frame size, model input, mean, std): the image is resized to frame size and JPEG-encoded
#   as the notebook's frame cache, then resized to the model input and normalized with mean / std
BACKBONES = {
    768: ("openai/clip-vit-base-patch16", 256, 224, CLIP_MEAN, CLIP_STD),          # ViT-B/16: EXP1, EXP5 (the paper's sizes)
    1024: ("openai/clip-vit-large-patch14-336", 384, 336, CLIP_MEAN, CLIP_STD),    # ViT-L/14-336: EXP6_3_2 ... EXP6_3_7 (same ratio)
    1280: ("facebook/dinov3-vith16plus-pretrain-lvd1689m", 384, 336, DINO_MEAN, DINO_STD),   # DINOv3 ViT-H+/16 (gated)
}

# ---------- the rest of the notebooks' configuration - the same for every backbone ----------
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 8, 0.4
LORA_TARGETS = {                           # q and v of every attention block (12 in ViT-B, 24 in ViT-L, 32 in ViT-H+)
    "clip": r".*vision_model.*\.(q_proj|v_proj)",           # CLIP: the vision tower only
    "dino": r".*layer\.\d+\.attention\.(q_proj|v_proj)",    # DINOv3's module names
}
BONA = 0                                   # label 0 = bona fide; the score is softmax[:, BONA]
THRESHOLD = 0.5                            # fixed decision threshold

# ---------- where the trained checkpoints live ----------
HF_REPO = "US10F/TPO_to_PID2"
HF_FILE = "exp6_3_7/best.pth"              # the default model: EXP6_3_7 (ViT-L/14-336), epoch 1
DOWNLOAD_DIR = "models"                    # downloads land in DOWNLOAD_DIR/<path in the repo>


def _linear_upcast(self, x):
    """nn.Linear.forward with the weight stored small (bfloat16, or int8 with a scale per output row):
    made float32 for this call only - float32 math, which CPUs without native bfloat16 run ~10x faster."""
    w = self.weight.to(x.dtype)
    if self.weight.dtype == torch.int8:
        w = w * self.weight_scale[:, None]
    return torch.nn.functional.linear(x, w, self.bias)


def load_lean(model_id, dtype, token=None):
    """AutoModel.from_pretrained for a one-file safetensors model, holding its Linear weights in `dtype`
    and everything else - biases, norms, layer scales, tokens, patch embedding, ~0.2 % of DINOv3 - in
    float32. dtype: torch.float32 (an ordinary model), torch.bfloat16 (half the memory) or torch.int8
    (a quarter: symmetric, one scale per output row). The Linear layers still compute in float32
    (_linear_upcast), so only the stored weights are rounded.
    Memory: the model is built with its parameters on "meta" (nothing allocated or randomly initialized),
    each parameter is then allocated in its storage dtype and filled one tensor at a time with plain file
    reads - no memory map, whose pages pile up next to the converted copy. DINOv3 ViT-H+ (a 3.4 GB float32
    file) holds ~0.85 GB in int8, under Streamlit Community Cloud's 2.7 GB limit with room to run."""
    import json
    import struct
    import types

    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModel

    path = hf_hub_download(model_id, "model.safetensors", token=token)
    config = AutoConfig.from_pretrained(model_id, token=token)
    with init_empty_weights(include_buffers=False):    # buffers (RoPE's inv_freq) are computed for real
        model = AutoModel.from_config(config, dtype=torch.float32)
    targets = {}                                       # parameter name -> (its module, attribute)
    for prefix, module in model.named_modules():
        small = isinstance(module, torch.nn.Linear) and dtype != torch.float32
        for name, p in list(module.named_parameters(recurse=False)):
            store = dtype if small and name == "weight" else torch.float32
            module._parameters[name] = torch.nn.Parameter(torch.empty(p.shape, dtype=store), requires_grad=False)
            targets[f"{prefix}.{name}" if prefix else name] = (module, name)
        if small:
            if dtype == torch.int8:
                module.register_buffer("weight_scale", torch.empty(module.out_features), persistent=False)
            module.forward = types.MethodType(_linear_upcast, module)
    dtypes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    loaded = set()
    with open(path, "rb", buffering=0) as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            name = key if key in targets else f"model.{key}"   # transformers 5 moved DINOv3's blocks under .model
            if name not in targets:
                raise ValueError(f"{model_id}: weight {key!r} has no place in {type(model).__name__}")
            begin, end = meta["data_offsets"]
            f.seek(8 + n + begin)
            t = torch.frombuffer(bytearray(f.read(end - begin)), dtype=dtypes[meta["dtype"]]).reshape(meta["shape"])
            module, attr = targets[name]
            with torch.no_grad():
                if module._parameters[attr].dtype == torch.int8:
                    t = t.float()
                    scale = t.abs().amax(dim=1).clamp(min=1e-12) / 127
                    module._parameters[attr].copy_(torch.round(t / scale[:, None]))
                    module.weight_scale.copy_(scale)
                else:
                    module._parameters[attr].copy_(t)
            loaded.add(name)
            if hasattr(os, "posix_fadvise"):           # Linux: drop the file's pages from the cache as we go
                os.posix_fadvise(f.fileno(), 8 + n + begin, end - begin, os.POSIX_FADV_DONTNEED)
    if loaded != set(targets):
        raise ValueError(f"{model_id}: {len(set(targets) - loaded)} weights missing from its file, "
                         f"e.g. {sorted(set(targets) - loaded)[:2]}")
    return model


def lora_width(sd):
    """The backbone width a FoundPAD state dict was trained on (its LoRA A matrices' input size,
    a BACKBONES key); ValueError when it is not one FoundPAD state dict."""
    widths = {v.shape[1] for k, v in sd.items() if "lora_A" in k}
    if len(widths) != 1 or next(iter(widths)) not in BACKBONES:
        raise ValueError(f"not a FoundPAD checkpoint: LoRA width {sorted(widths)}, "
                         f"expected one of {sorted(BACKBONES)}")
    return widths.pop()


class FoundPADInference:
    def __init__(self, checkpoint=HF_FILE, device=None, threshold=THRESHOLD, repo=HF_REPO,
                 token=None, download_dir=DOWNLOAD_DIR, dino_dtype=torch.float32):
        """Rebuild the checkpoint's backbone (BACKBONES: CLIP or DINOv3) + rank-stabilized LoRA +
        linear head and load the checkpoint - a local file, or a file inside `repo`, downloaded
        into `download_dir` the first time. dino_dtype: how the frozen DINOv3's Linear weights are
        stored - float32 as the notebook evaluates (3.4 GB), torch.bfloat16 (1.7 GB) or torch.int8
        (0.85 GB, the Streamlit Cloud app) where memory is short (load_lean). The math is float32."""
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

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        try:
            width = lora_width(sd)
        except ValueError as e:
            raise ValueError(f"{checkpoint} does not match FoundPAD: {e}") from None
        self.clip_id, self.frame_size, self.input_size, mean, std = BACKBONES[width]
        self.kind = "dino" if width == 1280 else "clip"   # the notebook's attribute name = its key prefix

        if self.kind == "dino":                            # float32 weights, as the notebook, unless dino_dtype
            backbone = load_lean(self.clip_id, dino_dtype, token=token)
            embed_dim = backbone.config.hidden_size        # 1280: the head sits on pooler_output
        else:
            backbone = CLIPModel.from_pretrained(self.clip_id)
            embed_dim = backbone.config.projection_dim     # CLIP's projected image embedding
        backbone.requires_grad_(False)
        # ModuleDict names its children "clip." / "dino." and "head." - the notebooks' FoundPAD key names
        self.model = torch.nn.ModuleDict({
            self.kind: get_peft_model(backbone, LoraConfig(
                r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                target_modules=LORA_TARGETS[self.kind], use_rslora=True, bias="none")),
            "head": torch.nn.Linear(embed_dim, 2)})

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        lost = [k for k in missing if "lora_" in k or k.startswith("head.")]
        if unexpected or lost:                  # only the frozen CLIP may be absent from the file
            raise ValueError(f"{checkpoint} does not match FoundPAD: {len(lost)} LoRA/head weights "
                             f"missing, {len(unexpected)} unexpected (e.g. {(lost + unexpected)[:2]})")
        self.model.to(self.device).eval()
        self.epoch = ck.get("epoch") if isinstance(ck, dict) else None
        self.to_tensor = transforms.Compose([transforms.Resize((self.input_size, self.input_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])

    def preprocess(self, image):
        """One image -> a (1, 3, input_size, input_size) tensor, exactly as the notebook built its
        frames: upright by EXIF orientation -> RGB -> frame_size bicubic -> JPEG q95 (the notebook's
        frame cache) -> input_size -> the backbone's normalization. frame_size / input_size are the
        loaded backbone's (BACKBONES): 256 -> 224 for ViT-B/16, 384 -> 336 for ViT-L/14-336 and
        DINOv3 ViT-H+/16. An image
        already frame_size square is used as is (no EXIF turn, no JPEG), as the notebook does.

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
            if img.size != (self.frame_size, self.frame_size):   # the notebook's frame cache
                img = ImageOps.exif_transpose(img)         # turn phone photos upright (EXIF Orientation)
                buf = io.BytesIO()
                img.convert("RGB").resize((self.frame_size, self.frame_size), Image.BICUBIC).save(buf, "JPEG", quality=95)
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
        n = self.input_size
        if not torch.is_tensor(x) or x.ndim != 4 or tuple(x.shape[1:]) != (3, n, n):
            raise ValueError(f"expected a (N, 3, {n}, {n}) tensor from preprocess(), "
                             f"got {tuple(x.shape) if torch.is_tensor(x) else type(x).__name__}")
        x = x.to(self.device)                        # float32, as the notebooks evaluate
        if self.kind == "dino":                      # DINOv3: the CLS token after the final norm
            e = self.model["dino"](pixel_values=x).pooler_output.float()
        else:
            e = self.model["clip"].get_image_features(pixel_values=x)
            if not torch.is_tensor(e):               # transformers >= 5 returns a ModelOutput
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
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), HF_FILE)   # the copy next to this file
    default = local if os.path.isfile(local) else HF_FILE
    ap.add_argument("--model", default=default,
                    help=f"a local .pth, or a file inside {HF_REPO} (default {default})")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args()
    pad = FoundPADInference(a.model, threshold=a.threshold)
    for path in a.images:
        try:
            r = pad.infer(pad.preprocess(path))[0]
            print(f"{r['decision']:9s}  P(bona fide) {r['score_bona_fide']:.4f}  {path}")
        except (FileNotFoundError, ValueError) as err:
            print(f"ERROR      {err}")
