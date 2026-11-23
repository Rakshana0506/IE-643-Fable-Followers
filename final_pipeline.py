"""
Full plant pipeline
 
  image
   └─ 1. DeepLabV3 (plant/weed mask)      -> binary mask of all plants (rcnn_test.py model)
       └─ 2. connected components -> keep masks whose area >= AREA_RATIO * max(mask area)
           └─ crop each kept mask (bounding box)
               └─ 3. Faster R-CNN         -> leaf bounding boxes inside each plant crop
                   └─ crop each leaf
                       ├─ 4. SR3 super-resolution -> DeepLab disease segmentation   (test.py)
                       └─ 5. DINOv2 classifier on EVERY leaf crop of a plant
                              -> majority vote per plant crop -> plant type + disease
                              (several plant crops from DeepLab are classified separately)

Imported by server.py (run_pipeline); needs index.html + Flask for the web interface.
"""
import json
import os
from pathlib import Path
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import functional as TF
 
# Use the PATCHED sr3_ft.py that was saved with the checkpoint (Drive/sr3/checkpoints/sr3_ft.py,
# it contains `_sample_ancestral`) and put it next to this script.
import sr3_files.sr3_ft as sr3read
from sr3_files.sr3_ft import build_unet, SR3Diffusion, square_crop_box, to_tensor as sr3_to_tensor
try:
    from sr3_files.sr3_ft import degrade as sr3_degrade     # the degradation SR3 was trained on: degrade(img, rng, LR)
except ImportError:
    sr3_degrade = None
 
_src = open(sr3read.__file__).read()
assert "_unet_forward_fp32_level" in _src and "_sample_ancestral" in _src, (
    f"{sr3read.__file__} is the unpatched sr3_ft.py. Copy Drive/sr3/checkpoints/sr3_ft.py "
    "(saved by the training notebook, contains the fp32 + ancestral patches) next to this script.")
 
 
class ResidualDiffusion(SR3Diffusion):
    """The model predicts (hr - lr_up) * res_scale."""
    def __init__(self, *a, res_scale=4.0, **k):
        super().__init__(*a, **k)
        self.res_scale = res_scale

    def _to_hr(self, r, lr_up):
        return (lr_up + r / self.res_scale).clamp(-1, 1)

    def _sample_ddim(self, model, lr_up, steps=100, eta=0.0, amp_dtype=None):
        r = SR3Diffusion._sample_ddim(self, model, lr_up, steps=steps, eta=eta, amp_dtype=amp_dtype)
        return self._to_hr(r, lr_up)

    def sample(self, model, lr_up, amp_dtype=None, **_):
        r = sr3read._sample_ancestral(self, model, lr_up, amp_dtype)
        return self._to_hr(r, lr_up)
 
# ==================================================
# 1. CONFIGURATION
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_dtype = (
    torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported()
    else torch.float32
)

# --- checkpoints (edit these) ---
PLANT_SEG_CKPT = r"plant_leaf_deeplab\best_deeplabv3_resnet50_plant.pth"      # stage 1: plant/weed mask (rcnn_test.py)
RCNN_CKPT = r"rcnn_files\fasterrcnn_best_new.pth"                         # stage 3: leaf detector
SR_CKPT = r"sr3_files\sr3_plantseg_residual.pth"                    # stage 4: SR3
SEG_CKPT = r"lesion_deeplab_files\deeplabv3plus_degaug_best.pth"             # stage 4: disease segmentation
DINO_CKPT = r"dinoV2_classifier\best.pt"                                     # stage 5: DINOv2 weights (OUT/best.pt from the notebook)
DINO_CLASSES_JSON = r"dinoV2_classifier\classes.json"                        # label -> id   (OUT/classes.json from the notebook)
DINO_CLASS_CROP_JSON = r"dinoV2_classifier\class_crop.json"                  # optional: label -> crop (see save snippet at bottom)

# --- stage 2: mask selection ---
FULL_MASK_FRAC = 0.85   # plant mask covers >= this fraction of the image (close-up) -> skip plant cropping, use the whole image
AREA_RATIO = 0.25      # keep masks with area >= AREA_RATIO * (largest mask area)
CROP_PADDING = 0.05    # extra margin around each mask bbox (fraction of bbox size)
MASK_BACKGROUND = False  # True: blank out pixels outside the mask before sending crop to R-CNN
MASK_LEAF_BACKGROUND = False  # True: every leaf crop keeps only plant-mask pixels, everything else is BLACK.
                              #   Used for degrade -> SR3 -> lesion model; lesions are kept inside the mask only and
                              #   leaf area (severity) = mask pixels. DINOv2 still gets the unmasked crop.
MASK_MIN_BG_FRAC = 0.30       # background is removed only if the background covers AT LEAST this fraction of the leaf crop;
                              #   below that the crop stays untouched (little background to remove, a mask error would cost lesions)
FILL_MASK_HOLES = True        # fill holes inside the plant mask first, so lesions the plant model left out (brown/necrotic
                              #   patches fully surrounded by leaf) are not blacked out
MIN_MASK_FRAC = 0.05          # if the mask covers less than this fraction of a leaf crop, the mask is ignored for that leaf

# --- other stages ---
DETECTION_THRESHOLD = 0.15
FALLBACK_TO_PLANT_CROP = True   # if R-CNN finds no leaf in a plant crop, treat the whole plant crop as one leaf
DETECT_MODE = "full_image"   # "full_image": run R-CNN on the whole photo (how it was trained), keep leaves
                             #               whose centre lies inside a kept plant crop
                             # "plant_crop": run R-CNN on each plant crop separately
RCNN_MIN_SIZE = None         # e.g. 800/1333 = torchvision default; set to what you trained with, None = leave as saved
SR_BATCH_SIZE = 16
DEGRADE_MODE = "none"        # how the normal 256x256 crop is degraded BEFORE SR3 (video_pipeline.py sets this):
                             #   "none"   : plain down-scale to the SR3 low-res size (original behaviour)
                             #   "sr3_ft" : sr3_ft.degrade, the random blur/noise/compression SR3 was trained with
                             #   "simple" : built-in blur + noise + JPEG + down-scale (if sr3_ft.degrade is unavailable)
DEGRADE_SEED = 0             # per-leaf seeds are DEGRADE_SEED + leaf number, so runs are reproducible
SEG_BATCH_SIZE = 16
HEALTHY_PCT = 30.0     # DINOv2 disease probability (%) below this -> "healthy"
LEAF_NMS_IOU = 0.3     # leaf boxes overlapping more than this IoU are merged (keep the higher score)
LEAF_CONTAIN = 0.8     # drop a box if >= this fraction of its area lies inside a higher-scoring box
DINO_BATCH_SIZE = 32
DINO_IMG_SIZE = 280
DINO_ARCH = "vit_small_patch14_dinov2.lvd142m"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==================================================
# 2. MODEL LOADING
# ==================================================
class _TVWrap(nn.Module):
    """torchvision DeepLab returns {"out": logits}; expose plain logits like smp does."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)["out"]


def build_deeplab(state_dict, num_classes=2):
    """
    Builds the DeepLab that matches the checkpoint's key layout:
      encoder.* / decoder.aspp.* / segmentation_head.*  -> segmentation_models_pytorch DeepLabV3Plus (resnet50)
      backbone.* / classifier.*                         -> torchvision deeplabv3_resnet50
    Always returns a module mapping (B,3,H,W) -> logits (B,num_classes,H,W).
    """
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    if any(k.startswith("encoder.") for k in state_dict):
        import segmentation_models_pytorch as smp          # pip install segmentation-models-pytorch
        model = smp.DeepLabV3Plus("resnet50", encoder_weights=None, classes=num_classes)
        model.load_state_dict(state_dict)
        return model
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier.")}
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes, aux_loss=False)
    model.load_state_dict(state_dict)
    return _TVWrap(model)


def load_plant_segmenter():
    ckpt = torch.load(PLANT_SEG_CKPT, map_location="cpu", weights_only=False)
    model = build_deeplab(ckpt["model_state_dict"]).to(device).eval()
    return model, ckpt.get("image_size", 512)


def load_detector():
    ckpt = torch.load(RCNN_CKPT, map_location=device, weights_only=True)
    det = fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, num_classes=ckpt["num_classes"]
    )
    det.load_state_dict(ckpt["model_state_dict"])
    return det.to(device).eval()


def load_sr3():
    # weights_only=False: the checkpoint holds a config dict + optimizer state, not just tensors
    ckpt = torch.load(SR_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    hr, lr, context = cfg["HR_SIZE"], cfg["LR_SIZE"], cfg["CONTEXT"]

    # UNet architecture comes from the checkpoint's own config (same as sr3_ft.build_sr3_from_checkpoint)
    ema = build_unet(cfg["UNET"], dropout=0.0, image_size=hr)
    ema.load_state_dict(ckpt["ema"], strict=True)
    ema.to(device).eval()

    args = (cfg["T"], cfg["BETA_START"], cfg["BETA_END"], device)
    if cfg.get("RES_SCALE"):                      # residual checkpoint (sr3_plantseg_residual.pth)
        diffusion = ResidualDiffusion(*args, res_scale=cfg["RES_SCALE"])
    else:                                         # plain checkpoint (sr3_plantseg_finetuned*.pth)
        diffusion = SR3Diffusion(*args)
    print("Loaded SR3 EMA step:", ckpt["step"], "| residual:", bool(cfg.get("RES_SCALE")))
    return ema, diffusion, hr, lr, context


def load_disease_segmenter():
    ckpt = torch.load(SEG_CKPT, map_location="cpu", weights_only=False)
    model = build_deeplab(ckpt["model_state_dict"]).to(device).eval()
    size = ckpt.get("crop_size", 520)
    mean = torch.tensor(ckpt.get("mean", IMAGENET_MEAN), device=device).view(1, 3, 1, 1)
    std = torch.tensor(ckpt.get("std", IMAGENET_STD), device=device).view(1, 3, 1, 1)
    return model, size, mean, std


def _crop_of_label(label):
    """Fallback when class_crop.json is missing: crop = first word(s) of the label."""
    two_word = ("bell pepper", "sweet potato", "sweet corn")
    for t in two_word:
        if label.startswith(t + " "):
            return t
    return label.split(" ")[0]


def load_dino():
    cid = json.loads(Path(DINO_CLASSES_JSON).read_text())      # label -> id
    n = len(cid)
    id2label = {i: c for c, i in cid.items()}

    if Path(DINO_CLASS_CROP_JSON).exists():
        label2crop = json.loads(Path(DINO_CLASS_CROP_JSON).read_text())
    else:
        print(f"[warn] {DINO_CLASS_CROP_JSON} not found - guessing crop from the label text")
        label2crop = {c: _crop_of_label(c) for c in cid}

    crops = sorted(set(label2crop[id2label[i]] for i in range(n)))
    # (n_classes x n_crops) one-hot matrix to sum class probabilities per crop
    class_to_crop = torch.zeros(n, len(crops))
    for i in range(n):
        class_to_crop[i, crops.index(label2crop[id2label[i]])] = 1.0

    model = timm.create_model(DINO_ARCH, pretrained=False, num_classes=n, img_size=DINO_IMG_SIZE)
    model.load_state_dict(torch.load(DINO_CKPT, map_location="cpu"))
    model.to(device).eval()
    return model, id2label, crops, class_to_crop.to(device)


# ==================================================
# 3. STAGE 1+2: PLANT MASK -> AREA-FILTERED CROPS
# ==================================================
@torch.no_grad()
def predict_plant_mask(image, model, size):
    """Binary plant/weed mask at the original image resolution (H, W) bool."""
    x = TF.to_tensor(image.resize((size, size), Image.Resampling.BILINEAR))
    x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)
    logits = model(x)
    logits = F.interpolate(logits, size=(image.height, image.width),
                           mode="bilinear", align_corners=False)
    return (logits.argmax(1)[0] == 1).cpu().numpy()


def crop_masks_by_area(image, mask):
    """
    Split `mask` into connected components, keep those whose area is at least
    AREA_RATIO * max(component area), and return one crop per kept component.
    """
    labeled, n = ndimage.label(mask)
    if n == 0:
        return []
    areas = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    threshold = AREA_RATIO * areas.max()
    slices = ndimage.find_objects(labeled)

    out = []
    for comp_id, (area, sl) in enumerate(zip(areas, slices), start=1):
        if area < threshold:
            continue
        y1, y2, x1, x2 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        pad_x = int((x2 - x1) * CROP_PADDING)
        pad_y = int((y2 - y1) * CROP_PADDING)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(image.width, x2 + pad_x), min(image.height, y2 + pad_y)

        crop = image.crop((x1, y1, x2, y2))
        if MASK_BACKGROUND:
            m = (labeled[y1:y2, x1:x2] == comp_id)
            arr = np.array(crop)
            arr[~m] = 0
            crop = Image.fromarray(arr)

        out.append({"crop": crop, "box": (x1, y1, x2, y2), "mask_area": int(area)})
    return out


# ==================================================
# 4. STAGE 3: LEAF DETECTION INSIDE EACH PLANT CROP
# ==================================================
@torch.no_grad()
def detect_leaves(img, offset, detector):
    """Runs R-CNN on `img`; boxes are shifted by `offset` into original-image coordinates."""
    ox, oy = offset
    pred = detector([TF.to_tensor(img).to(device)])[0]
    top = pred["scores"][:5].tolist()
    print(f"  R-CNN on {img.width}x{img.height}: {len(pred['scores'])} raw boxes, top scores {[round(t, 2) for t in top]}")

    keep = pred["scores"] >= DETECTION_THRESHOLD
    boxes, scores, labels = pred["boxes"][keep], pred["scores"][keep], pred["labels"][keep]

    # overlap handling: IoU NMS, then drop boxes mostly contained in a better box
    idx = torchvision.ops.nms(boxes, scores, LEAF_NMS_IOU)
    selected = []
    for i in idx.tolist():                       # idx is sorted by score, best first
        bx = boxes[i]
        area = ((bx[2] - bx[0]) * (bx[3] - bx[1])).clamp(min=1e-6)
        contained = False
        for j in selected:
            ox1, oy1 = torch.max(bx[0], boxes[j][0]), torch.max(bx[1], boxes[j][1])
            ox2, oy2 = torch.min(bx[2], boxes[j][2]), torch.min(bx[3], boxes[j][3])
            inter = (ox2 - ox1).clamp(min=0) * (oy2 - oy1).clamp(min=0)
            if inter / area >= LEAF_CONTAIN:
                contained = True
                break
        if not contained:
            selected.append(i)

    leaves = []
    for i in selected:
        x1, y1, x2, y2 = boxes[i].cpu().tolist()
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img.width, int(x2)), min(img.height, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        leaves.append({
            "crop": img.crop((x1, y1, x2, y2)),
            "label": int(labels[i].item()),
            "score": scores[i].item(),
            "box": (x1 + ox, y1 + oy, x2 + ox, y2 + oy),   # coordinates in the ORIGINAL image
        })
    return leaves


# ==================================================
# 5. STAGE 4: LEAF CROPS (256x256) -> SR3 -> LESION MASKS -> SEVERITY
# ==================================================
def overlay_mask(img, mask, color=(255, 0, 0), alpha=0.5):
    """Paints `mask` (bool, same size as img) on top of `img`."""
    arr = np.array(img.convert("RGB")).astype(np.float32)
    arr[mask] = (1 - alpha) * arr[mask] + alpha * np.array(color, np.float32)
    return Image.fromarray(arr.astype(np.uint8))


def prepare_leaf_crops(leaves, hr):
    """Every leaf crop -> hr x hr (256x256)."""
    for l in leaves:
        raw = l["crop"].resize((hr, hr), Image.Resampling.LANCZOS)
        l["img_raw"] = raw                                   # unmasked, what DINOv2 sees
        m = l.get("leaf_mask")
        l["mask256"] = None
        if m is not None and FILL_MASK_HOLES:
            m = ndimage.binary_fill_holes(m)
        if (MASK_LEAF_BACKGROUND and m is not None and m.mean() >= MIN_MASK_FRAC
                and (1.0 - m.mean()) >= MASK_MIN_BG_FRAC):
            m256 = resize_mask(m, hr, hr)
            arr = np.array(raw)
            arr[~m256] = 0                                   # everything outside the plant mask -> black
            l["mask256"] = m256
            l["img_n"] = Image.fromarray(arr)
        else:
            l["img_n"] = raw
    return leaves


SAMPLERS = {
    "ancestral": "Ancestral (checkpoint default; step count is fixed by the sampler)",
    "ddim": "DDIM, deterministic (eta = 0)",
    "ddim_stochastic": "DDIM, stochastic (eta = 1, ancestral-style on N steps)",
}
SR_AMP = torch.bfloat16 if device.type == "cuda" else None


def sr_sample(diffusion, ema, cond, sampler, steps):
    if sampler == "ancestral":
        return diffusion.sample(ema, cond, amp_dtype=SR_AMP)
    eta = 0.0 if sampler == "ddim" else 1.0
    return diffusion._sample_ddim(ema, cond, steps=int(steps), eta=eta, amp_dtype=SR_AMP)


def simple_degrade(img, rng, lr):
    """Fallback degradation: Gaussian blur, noise, JPEG, then down-scale to the SR3 low-res size."""
    import io
    from PIL import ImageFilter
    img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 2.5))))
    arr = np.asarray(img).astype(np.float32) + rng.normal(0, float(rng.uniform(1, 6)), (img.height, img.width, 3))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(rng.integers(40, 90)))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB").resize((lr, lr), Image.Resampling.BICUBIC)


def degrade_crop(img_n, hr, lr, seed):
    """Degraded version of the normal crop at hr x hr: this is both the SR3 condition and the 'degraded input'."""
    rng = np.random.default_rng(seed)
    if DEGRADE_MODE == "sr3_ft" and sr3_degrade is not None:
        low = sr3_degrade(img_n, rng, lr)
    elif DEGRADE_MODE in ("sr3_ft", "simple"):
        low = simple_degrade(img_n, rng, lr)
    else:
        low = img_n.resize((lr, lr), Image.Resampling.LANCZOS)
    return low, low.resize((hr, hr), Image.Resampling.BICUBIC)      # (low-res degraded, bicubic up-sampled to hr)


@torch.no_grad()
def super_resolve_crops(items, ema, diffusion, hr, lr, sampler="ancestral", steps=100):
    """normal 256x256 crop -> degrade -> (bicubic up to 256) = SR3 condition -> SR3 reconstruction 256x256."""
    for n, it in enumerate(items):
        it["img_low"], it["img_d"] = degrade_crop(it["img_n"], hr, lr, DEGRADE_SEED + n)
        # same degradation (same seed) of the UNMASKED crop: only used to fill the background when saving results,
        # so the saved degraded / bicubic images show a background that is blurred like the leaf, with no visible mask
        it["img_low_full"], it["img_d_full"] = (degrade_crop(it["img_raw"], hr, lr, DEGRADE_SEED + n)
                                                if it.get("mask256") is not None else (it["img_low"], it["img_d"]))
    for s in range(0, len(items), SR_BATCH_SIZE):
        batch_items = items[s:s + SR_BATCH_SIZE]
        cond = torch.stack([sr3_to_tensor(it["img_d"]) for it in batch_items]).to(device)   # [-1, 1]
        sr = sr_sample(diffusion, ema, cond, sampler, steps)
        sr = ((sr.float().clamp(-1, 1) + 1) / 2).cpu()                                 # -> [0, 1]
        for it, t in zip(batch_items, sr):
            it["img_s"] = TF.to_pil_image(t)
    return items


@torch.no_grad()
def predict_disease_mask(img, seg, seg_size, mean, std):
    """Lesion mask (bool, H x W of `img`) from the DeepLabV3+ disease model."""
    x = TF.to_tensor(img).unsqueeze(0).to(device)
    x = F.interpolate(x, size=(seg_size, seg_size), mode="bilinear", align_corners=False)
    logits = seg((x - mean) / std)
    logits = F.interpolate(logits, size=(img.height, img.width), mode="bilinear", align_corners=False)
    return (logits.argmax(1)[0] == 1).cpu().numpy()


def severity_pct(lesion, leaf=None):
    """Lesion area as a % of the 256x256 leaf crop (or of the leaf-mask pixels when a mask is given)."""
    if leaf is not None:
        return 100.0 * (lesion & leaf).sum() / max(int(leaf.sum()), 1)
    return 100.0 * lesion.mean()


def segment_and_score(items, seg, seg_size, mean, std):
    for it in items:
        for tag in ("n", "d", "s"):          # n = normal crop, d = degraded input (bicubic up-sampled), s = SR3
            les = predict_disease_mask(it[f"img_{tag}"], seg, seg_size, mean, std)
            if it.get("mask256") is not None:
                les = les & it["mask256"]                    # lesions only inside the plant mask (no soil / black bg)
            it[f"lesion_{tag}"] = les
            it[f"sev_{tag}"] = severity_pct(les, it.get("mask256"))
        both = (it["lesion_n"] | it["lesion_s"]).sum()
        it["mask_iou"] = float((it["lesion_n"] & it["lesion_s"]).sum() / both) if both else 1.0
        bd = (it["lesion_n"] | it["lesion_d"]).sum()
        it["mask_iou_d"] = float((it["lesion_n"] & it["lesion_d"]).sum() / bd) if bd else 1.0
    return items


def resize_mask(m, w, h):
    return np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127


def total_severity(leaves, image_size):
    """
    Whole-image severity = total lesion area / total leaf area over ALL leaves.
    Each leaf's lesion mask is pasted back into its box on a full-image canvas, so a pixel covered by
    two overlapping leaf boxes is counted once (union) in both the leaf area and the lesion area.
    """
    W, H = image_size
    leaf = np.zeros((H, W), bool)
    les = {t: np.zeros((H, W), bool) for t in ("n", "d", "s")}
    for l in leaves:
        x1, y1, x2, y2 = l["box"]
        if l.get("mask256") is not None:
            leaf[y1:y2, x1:x2] |= resize_mask(l["mask256"], x2 - x1, y2 - y1)   # leaf area = mask pixels only
        else:
            leaf[y1:y2, x1:x2] = True
        for t in ("n", "d", "s"):
            les[t][y1:y2, x1:x2] |= resize_mask(l[f"lesion_{t}"], x2 - x1, y2 - y1)
    leaf_px = max(int(leaf.sum()), 1)
    return {"leaf_px": leaf_px,
            "lesion_px": {t: int(les[t].sum()) for t in les},
            "severity": {t: 100.0 * les[t].sum() / leaf_px for t in les},
            "canvas": les}


# ==================================================
# 6. STAGE 5: DINOv2 ON EVERY LEAF CROP (majority per plant)
# ==================================================
@torch.no_grad()
def classify_original(image, dino, id2label, crops, class_to_crop):
    """
    DINOv2 prediction for ONE image (here: one 256x256 leaf crop).
      plant type      = crop whose classes hold the most probability
      disease prob    = 1 - P(healthy class | predicted plant type)
      disease prob < HEALTHY_PCT %  -> "healthy"
      otherwise                     -> most probable non-healthy class of that plant type
    """
    x = TF.normalize(
        TF.to_tensor(image.resize((DINO_IMG_SIZE, DINO_IMG_SIZE), Image.Resampling.BILINEAR)),
        IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = dino(x)
    p = logits.float().softmax(1)[0]
    crop_p = p @ class_to_crop
    ci = crop_p.argmax().item()

    members = class_to_crop[:, ci] > 0
    is_healthy = torch.tensor([id2label[i].endswith("healthy") for i in range(len(p))], device=p.device)
    member_mass = p[members].sum().clamp(min=1e-9)
    disease_prob = 100.0 * (1 - p[members & is_healthy].sum() / member_mass).item()

    top = p.masked_fill(~members | is_healthy, -1).argmax().item()      # best disease class of this plant
    if disease_prob < HEALTHY_PCT or p[members & ~is_healthy].numel() == 0:
        diagnosis = "healthy"
    else:
        diagnosis = id2label[top]
    return dict(plant_type=crops[ci], plant_type_conf=crop_p[ci].item(),
                disease_prob=disease_prob, diagnosis=diagnosis,
                top_class=id2label[p.argmax().item()], top_conf=p.max().item())


def majority(pairs):
    """pairs = [(value, confidence)]. Most common value; ties go to the higher summed confidence."""
    score = {}
    for v, c in pairs:
        n, tot = score.get(v, (0, 0.0))
        score[v] = (n + 1, tot + c)
    best = max(score, key=lambda v: score[v])
    return best, score[best][0]


def vote_leaves(results):
    """
    results = classify_original() output for every leaf crop of ONE plant crop.
    1) plant type = majority over its leaves
    2) diagnosis  = majority over the leaves that voted for that plant (labels of different crops never mix);
       each leaf already applies: DINOv2 disease prob < HEALTHY_PCT -> "healthy"
    """
    plant, pn = majority([(r["plant_type"], r["plant_type_conf"]) for r in results])
    sel = [r for r in results if r["plant_type"] == plant]
    diag, dn = majority([(r["diagnosis"], r["top_conf"]) for r in sel])
    agree = [r for r in sel if r["diagnosis"] == diag]
    mean = lambda xs: sum(xs) / len(xs)
    return dict(plant_type=plant, plant_conf=mean([r["plant_type_conf"] for r in sel]),
                plant_votes=(pn, len(results)), diagnosis=diag, diagnosis_votes=(dn, len(sel)),
                disease_prob=mean([r["disease_prob"] for r in agree]))


# ==================================================
# 7. MODELS (loaded once) AND THE FULL PIPELINE
# ==================================================
_M = {}


def get_models():
    if not _M:
        _M["plant_seg"], _M["plant_size"] = load_plant_segmenter()
        _M["detector"] = load_detector()
        if RCNN_MIN_SIZE:
            _M["detector"].transform.min_size = (RCNN_MIN_SIZE,)
        _M["ema"], _M["diffusion"], _M["HR"], _M["LR"], _M["CONTEXT"] = load_sr3()
        _M["dis_seg"], _M["dis_size"], _M["mean"], _M["std"] = load_disease_segmenter()
        _M["dino"], _M["id2label"], _M["crops"], _M["class_to_crop"] = load_dino()
    return _M


def run_pipeline(image, sampler="ancestral", steps=100, det_thresh=None, healthy_pct=None, single_plant=False):
    """
    image: PIL RGB. Returns result images (PIL), one record per plant crop (majority-vote plant + disease,
    severity), one record per leaf, and the whole-image severity.
    """
    global DETECTION_THRESHOLD, HEALTHY_PCT
    if det_thresh is not None:
        DETECTION_THRESHOLD = float(det_thresh)
    if healthy_pct is not None:
        HEALTHY_PCT = float(healthy_pct)

    import time
    M = get_models()
    image = image.convert("RGB")
    log = []

    # Stage 1: plant / weed mask
    mask = predict_plant_mask(image, M["plant_seg"], M["plant_size"])
    log.append(f"Plant mask covers {mask.mean() * 100:.1f}% of the image")

    # Stage 2: plant crops (skipped for close-ups)
    # single_plant=True: the image is already one plant crop (video pipeline), so no plant splitting
    plants = [] if (single_plant or mask.mean() >= FULL_MASK_FRAC) else crop_masks_by_area(image, mask)
    if not plants:
        log.append("Close-up / no separate plants: whole image used as the single plant crop")
        plants = [{"crop": image, "box": (0, 0, image.width, image.height), "mask_area": int(mask.sum())}]
    log.append(f"Plant crops: {len(plants)}")

    # Stage 3: leaves
    leaves = []
    if DETECT_MODE == "full_image":
        for l in detect_leaves(image, (0, 0), M["detector"]):
            cx, cy = (l["box"][0] + l["box"][2]) / 2, (l["box"][1] + l["box"][3]) / 2
            for pi, p in enumerate(plants):
                x1, y1, x2, y2 = p["box"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    l["plant_idx"] = pi
                    leaves.append(l)
                    break
    else:
        for pi, plant in enumerate(plants):
            for l in detect_leaves(plant["crop"], plant["box"][:2], M["detector"]):
                l["plant_idx"] = pi
                leaves.append(l)
    n_detected = len(leaves)
    covered_plants = {l["plant_idx"] for l in leaves}
    if FALLBACK_TO_PLANT_CROP:
        covered = {l["plant_idx"] for l in leaves}
        for pi, p in enumerate(plants):
            if pi not in covered:
                leaves.append({"crop": p["crop"], "label": -1, "score": 0.0, "box": p["box"], "plant_idx": pi})
    log.append(f"Leaves: {len(leaves)} ({n_detected} detected, {len(leaves) - n_detected} fallback plant crops)")

    # Stage 4: 256 crops (background black) -> degrade -> SR3 -> lesion masks
    for l in leaves:
        x1, y1, x2, y2 = l["box"]
        l["leaf_mask"] = mask[y1:y2, x1:x2]                  # plant mask inside this leaf box (original resolution)
    prepare_leaf_crops(leaves, M["HR"])
    t0 = time.time()
    super_resolve_crops(leaves, M["ema"], M["diffusion"], M["HR"], M["LR"], sampler, steps)
    sr_secs = time.time() - t0
    log.append(f"SR3 ({sampler}, {steps if sampler != 'ancestral' else 'fixed'} steps): {sr_secs:.1f}s for {len(leaves)} crops")
    for l in leaves:                                         # SR3 output: same black background as its input
        if l.get("mask256") is not None:
            a = np.array(l["img_s"]); a[~l["mask256"]] = 0; l["img_s"] = Image.fromarray(a)
    segment_and_score(leaves, M["dis_seg"], M["dis_size"], M["mean"], M["std"])
    total = total_severity(leaves, image.size)

    # Stage 5: DINOv2 on every leaf crop, then a majority vote PER PLANT CROP
    for l in leaves:
        l["dino"] = classify_original(l["img_raw"], M["dino"], M["id2label"], M["crops"], M["class_to_crop"])
    plant_results = []
    for pi, p in enumerate(plants):
        ids = [i for i, l in enumerate(leaves) if l["plant_idx"] == pi]
        if not ids:
            continue
        mine = [leaves[i] for i in ids]
        v = vote_leaves([l["dino"] for l in mine])
        sev = total_severity(mine, image.size)
        thumb = p["crop"].copy()
        thumb.thumbnail((480, 480))
        plant_results.append(dict(
            idx=pi + 1, box=p["box"], thumb=thumb, leaf_ids=[i + 1 for i in ids], n_leaves=len(ids),
            from_detection=pi in covered_plants,      # False: no leaf found, the whole plant crop was classified
            plant_type=v["plant_type"], plant_conf=v["plant_conf"], plant_votes=v["plant_votes"],
            diagnosis=v["diagnosis"], diagnosis_votes=v["diagnosis_votes"], disease_prob=v["disease_prob"],
            healthy=v["diagnosis"] == "healthy",
            sev_n=sev["severity"]["n"], sev_d=sev["severity"]["d"], sev_s=sev["severity"]["s"],
            lesion_px_n=sev["lesion_px"]["n"], lesion_px_s=sev["lesion_px"]["s"], leaf_px=sev["leaf_px"],
        ))
        r = plant_results[-1]
        log.append(f"Plant {r['idx']}: {r['plant_type']} ({r['plant_votes'][0]}/{r['plant_votes'][1]} leaves), "
                   f"{r['diagnosis']} ({r['diagnosis_votes'][0]}/{r['diagnosis_votes'][1]} leaves), "
                   f"severity {r['sev_n']:.1f}% normal / {r['sev_s']:.1f}% SR3")

    # Images
    plant_img = overlay_mask(image, mask, color=(0, 255, 0), alpha=0.4)
    boxes_img = image.copy()
    d = ImageDraw.Draw(boxes_img)
    lw = max(2, image.width // 400)
    try:
        font = ImageFont.load_default(size=max(14, image.width // 60))
    except TypeError:                                  # older Pillow
        font = ImageFont.load_default()
    for pi, p in enumerate(plants, 1):
        d.rectangle(p["box"], outline=(0, 120, 255), width=lw * 2)
        d.text((p["box"][0] + 6, p["box"][1] + 6), f"P{pi}", fill=(0, 120, 255), font=font)
    for i, l in enumerate(leaves, 1):
        d.rectangle(l["box"], outline=(0, 255, 0), width=lw)
        d.text((l["box"][0] + 3, l["box"][1] + 3), str(i), fill=(255, 255, 0), font=font)

    return {
        "log": log, "plants": plant_results, "n_leaves": len(leaves), "total": total, "sr_seconds": sr_secs,
        "plant_mask_img": plant_img, "boxes_img": boxes_img,
        "lesions_normal_img": overlay_mask(image, total["canvas"]["n"]),
        "lesions_sr3_img": overlay_mask(image, total["canvas"]["s"]),
        "lesions_bicubic_img": overlay_mask(image, total["canvas"]["d"]),
        "leaves": [{
            "idx": i, "plant_idx": l["plant_idx"] + 1, "score": l["score"],
            "sev_n": l["sev_n"], "sev_s": l["sev_s"], "iou": l["mask_iou"],
            "sev_d": l["sev_d"], "iou_d": l["mask_iou_d"],
            "plant": l["dino"]["plant_type"], "diagnosis": l["dino"]["diagnosis"], "disease_prob": l["dino"]["disease_prob"],
            "plant_conf": l["dino"]["plant_type_conf"], "top_conf": l["dino"]["top_conf"],
            # raw material for the video pipeline's saved results (server.py ignores these keys)
            "box": l["box"], "img_n": l["img_n"], "img_raw": l["img_raw"], "mask256": l["mask256"], "img_d": l["img_d"], "img_low": l["img_low"], "img_low_full": l["img_low_full"], "img_d_full": l["img_d_full"], "img_s": l["img_s"], "lesion_n": l["lesion_n"], "lesion_d": l["lesion_d"], "lesion_s": l["lesion_s"],
            "overlay_n": overlay_mask(l["img_n"], l["lesion_n"]), "overlay_d": overlay_mask(l["img_d"], l["lesion_d"]), "overlay_s": overlay_mask(l["img_s"], l["lesion_s"]),
        } for i, l in enumerate(leaves, 1)],
    }
