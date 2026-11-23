"""
Analysis layer for the web UI (server.py): runs final_pipeline on an IMAGE or a VIDEO and writes, per plant,
the images and the numbers that index.html shows.

  image : DeepLab plant masks -> one result per plant crop
  video : sample frames -> detect + track plants -> ONE best view per plant (sharp, large, not cut by the frame edge)

Per leaf the result images are: original, degraded (low-res), bicubic, SR3, lesions on original / bicubic / SR3,
bicubic vs original, SR3 vs original (yellow = both, blue = only on the original, red = not on the original).
The plant-level images are the same overlays on the whole plant. No mask images are written: the background
mask is applied inside the pipeline only.

Numbers (original crop = reference), for bicubic and for SR3: IoU, overlap area, new-lesion area, lost area,
lesion counts (kept / lost / new / shifted).
"""
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

import final_pipeline as fp
from lesion_compare import agreement_overlay, compare_lesions, show_unmasked
from plant_tracker import PlantTracker, estimate_motion, pick_frames, sharpness

DEFAULTS = dict(sampler="ddim", steps=50, degrade_mode="sr3_ft", mask_background=True, mask_min_bg_frac=0.30,
                fill_holes=True, healthy_pct=30.0, det_thresh=0.15, sample_fps=3.0)
PHOTO_AREA_RATIO = fp.AREA_RATIO          # plant-blob filter for photos
VIDEO_AREA_RATIO = 0.05                   # lower for video frames (same value as video_pipeline.py)
MIN_PLANT_AREA_FRAC, BLUR_FRACTION = 0.003, 0.5
MIN_OBSERVATIONS, MAX_MISSED, IOU_MIN = 3, 4, 0.15


# ---------------------------------------------------------------- small helpers
def configure(cfg):
    fp.DEGRADE_MODE, fp.DEGRADE_SEED = cfg["degrade_mode"], 0
    fp.MASK_LEAF_BACKGROUND = bool(cfg["mask_background"])
    fp.MASK_MIN_BG_FRAC = float(cfg["mask_min_bg_frac"])
    fp.FILL_MASK_HOLES = bool(cfg["fill_holes"])


class Writer:
    """Saves images under the job folder and returns the URL the page uses."""
    def __init__(self, out_dir, url_base):
        self.out_dir, self.url_base = Path(out_dir), url_base

    def __call__(self, img, rel, png=False):
        path = self.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        img = img.convert("RGB")
        if png:
            img.save(path.with_suffix(".png"))
            rel = str(Path(rel).with_suffix(".png"))
        else:
            img.save(path.with_suffix(".jpg"), quality=90)
            rel = str(Path(rel).with_suffix(".jpg"))
        return f"{self.url_base}/{rel.replace(chr(92), '/')}"


def pixel_metrics(orig, found, area):
    """Pixel agreement between the lesion mask on the original crop and on bicubic / SR3 (bool masks, same grid)."""
    o, x = orig.astype(bool), found.astype(bool)
    inter, union = int((o & x).sum()), int((o | x).sum())
    area = max(int(area), 1)
    return dict(
        iou=(inter / union) if union else 1.0,
        original_px=int(o.sum()), found_px=int(x.sum()), overlap_px=inter,
        new_px=int((x & ~o).sum()), lost_px=int((o & ~x).sum()), leaf_px=area,
        overlap_pct_of_original=(100.0 * inter / o.sum()) if o.sum() else None,
        overlap_pct_of_leaf=100.0 * inter / area,
        new_pct_of_leaf=100.0 * int((x & ~o).sum()) / area,
        lost_pct_of_leaf=100.0 * int((o & ~x).sum()) / area,
    )


def with_counts(pm, st):
    return dict(pm, lesions_original=st["lesions_normal"], lesions_found=st["lesions_sr3"], kept=st["kept"],
                shifted=st["shifted"], lost=st["lost_in_sr3"], new=st["new_in_sr3"])


def combine(metric_list):
    """Add up metrics of several leaves / plants (areas and counts add; IoU is recomputed from the sums)."""
    keys = ("original_px", "found_px", "overlap_px", "new_px", "lost_px", "leaf_px",
            "lesions_original", "lesions_found", "kept", "shifted", "lost", "new")
    s = {k: sum(m.get(k, 0) for m in metric_list) for k in keys}
    union = s["original_px"] + s["found_px"] - s["overlap_px"]
    area = max(s["leaf_px"], 1)
    s.update(iou=(s["overlap_px"] / union) if union else 1.0,
             overlap_pct_of_original=(100.0 * s["overlap_px"] / s["original_px"]) if s["original_px"] else None,
             overlap_pct_of_leaf=100.0 * s["overlap_px"] / area, new_pct_of_leaf=100.0 * s["new_px"] / area,
             lost_pct_of_leaf=100.0 * s["lost_px"] / area)
    return s


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------- one plant
def build_plant(p, res, crop, image_size, write, pid):
    """p: plant record from run_pipeline, crop: PIL of the plant, image_size: size of the image run_pipeline saw."""
    mine = [l for l in res["leaves"] if l["plant_idx"] == p["idx"]]
    base = f"plant_{pid:02d}"
    x1, y1, x2, y2 = (int(v) for v in p["box"])
    tot = fp.total_severity(mine, image_size)
    cn, cd, cs = (tot["canvas"][k][y1:y2, x1:x2] for k in ("n", "d", "s"))
    crop = crop.convert("RGB")

    leaves = []
    for l in mine:
        n = f"{base}/leaf_{l['idx']:02d}"
        orig = l["img_raw"]
        m = l["mask256"]
        low = show_unmasked(l["img_low"], l["img_low_full"], m)       # background put back for display only
        bic = show_unmasked(l["img_d"], l["img_d_full"], m)
        srx = show_unmasked(l["img_s"], orig, m)
        st_s, mn, ms = compare_lesions(l["lesion_n"], l["lesion_s"])
        st_b, mn_b, mb = compare_lesions(l["lesion_n"], l["lesion_d"])
        area = int(m.sum()) if m is not None else orig.size[0] * orig.size[1]
        leaves.append(dict(
            idx=l["idx"], score=l["score"], plant=l["plant"], diagnosis=l["diagnosis"],
            disease_prob=l["disease_prob"], sev_original=l["sev_n"], sev_bicubic=l["sev_d"], sev_sr3=l["sev_s"],
            images=dict(
                original=write(orig, f"{n}_1_original"),
                degraded=write(low, f"{n}_2_degraded", png=True),
                bicubic=write(bic, f"{n}_3_bicubic"),
                sr3=write(srx, f"{n}_4_sr3"),
                lesions_original=write(fp.overlay_mask(orig, l["lesion_n"]), f"{n}_5_lesions_original"),
                lesions_bicubic=write(fp.overlay_mask(bic, l["lesion_d"]), f"{n}_6_lesions_bicubic"),
                lesions_sr3=write(fp.overlay_mask(srx, l["lesion_s"]), f"{n}_7_lesions_sr3"),
                bicubic_vs_original=write(agreement_overlay(orig, mn_b, mb), f"{n}_8_bicubic_vs_original"),
                sr3_vs_original=write(agreement_overlay(orig, mn, ms), f"{n}_9_sr3_vs_original"),
            ),
            metrics=dict(bicubic=with_counts(pixel_metrics(l["lesion_n"], l["lesion_d"], area), st_b),
                         sr3=with_counts(pixel_metrics(l["lesion_n"], l["lesion_s"], area), st_s)),
        ))

    # counts add up over the leaves; areas / IoU come from the canvases (every leaf pasted back on the plant,
    # overlapping leaf boxes counted once)
    metrics = {}
    for key, c in (("bicubic", cd), ("sr3", cs)):
        metrics[key] = {**combine([lf["metrics"][key] for lf in leaves]), **pixel_metrics(cn, c, tot["leaf_px"])}

    return dict(
        id=pid, box=[x1, y1, x2, y2], n_leaves=len(mine), from_detection=bool(p["from_detection"]),
        plant_type=p["plant_type"], plant_conf=p["plant_conf"], plant_votes=list(p["plant_votes"]),
        diagnosis=p["diagnosis"], diagnosis_votes=list(p["diagnosis_votes"]), disease_prob=p["disease_prob"],
        healthy=bool(p["healthy"]),
        severity=dict(original=p["sev_n"], bicubic=p["sev_d"], sr3=p["sev_s"]),
        images=dict(
            crop=write(crop, f"{base}/plant_0_crop"),
            lesions_original=write(fp.overlay_mask(crop, cn), f"{base}/plant_1_lesions_original"),
            lesions_bicubic=write(fp.overlay_mask(crop, cd), f"{base}/plant_2_lesions_bicubic"),
            lesions_sr3=write(fp.overlay_mask(crop, cs), f"{base}/plant_3_lesions_sr3"),
            bicubic_vs_original=write(agreement_overlay(crop, cn, cd), f"{base}/plant_4_bicubic_vs_original"),
            sr3_vs_original=write(agreement_overlay(crop, cn, cs), f"{base}/plant_5_sr3_vs_original"),
        ),
        metrics=metrics, leaves=leaves,
    )


def finish(plants, mode, cfg, seconds, extra=None):
    allm = lambda arm: combine([p["metrics"][arm] for p in plants]) if plants else None
    out = dict(mode=mode, settings=cfg, seconds=seconds, n_plants=len(plants),
               n_leaves=sum(p["n_leaves"] for p in plants), plants=plants,
               totals=dict(bicubic=allm("bicubic"), sr3=allm("sr3")))
    out.update(extra or {})
    return out


# ---------------------------------------------------------------- image
def analyse_image(image, cfg, out_dir, url_base, progress):
    t0 = time.time()
    configure(cfg)
    fp.AREA_RATIO = PHOTO_AREA_RATIO
    write = Writer(out_dir, url_base)
    image = image.convert("RGB")
    progress("Plant masks, leaf detection, degrade, SR3, lesions, DINOv2 (this is the slow step)", 0.1)
    res = fp.run_pipeline(image, sampler=cfg["sampler"], steps=int(cfg["steps"]), det_thresh=cfg["det_thresh"],
                          healthy_pct=cfg["healthy_pct"])
    progress("Building results", 0.9)
    plants = []
    for p in res["plants"]:
        crop = image.crop(tuple(int(v) for v in p["box"]))
        plants.append(build_plant(p, res, crop, image.size, write, len(plants) + 1))

    overview = image.copy()                       # the photo with every plant box (plant masks are not shown)
    d = ImageDraw.Draw(overview)
    lw, font = max(3, image.width // 300), _font(max(16, image.width // 50))
    for pl in plants:
        d.rectangle(pl["box"], outline=(0, 120, 255), width=lw)
        d.text((pl["box"][0] + 8, pl["box"][1] + 8), f"Plant {pl['id']}", fill=(255, 255, 0), font=font)
    return finish(plants, "image", cfg, time.time() - t0,
                  dict(overview=write(overview, "overview"), log=res["log"]))


# ---------------------------------------------------------------- video
@torch.no_grad()
def _embed(dino, img):
    x = TF.normalize(TF.to_tensor(img.resize((fp.DINO_IMG_SIZE, fp.DINO_IMG_SIZE))),
                     fp.IMAGENET_MEAN, fp.IMAGENET_STD).unsqueeze(0).to(fp.device)
    f = dino.forward_head(dino.forward_features(x), pre_logits=True)[0].float()
    return F.normalize(f, dim=0).cpu().numpy()


def analyse_video(path, cfg, out_dir, url_base, progress):
    t0 = time.time()
    configure(cfg)
    fp.AREA_RATIO = VIDEO_AREA_RATIO
    write = Writer(out_dir, url_base)
    frames_dir = Path(out_dir) / "_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("That video could not be opened.")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, round(native_fps / float(cfg["sample_fps"])))
    progress(f"Reading the video ({native_fps:.0f} fps, sampling every {every} frames)", 0.02)
    sampled, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            fpth = frames_dir / f"{len(sampled):05d}.jpg"
            cv2.imwrite(str(fpth), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            sampled.append((i, fpth, sharpness(cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))))
        i += 1
    cap.release()
    if not sampled:
        raise ValueError("No frames could be read from that video.")
    median_sharp = float(np.median([s[2] for s in sampled]))

    M = fp.get_models()
    tracker = PlantTracker(iou_min=IOU_MIN, max_missed=MAX_MISSED, min_obs=MIN_OBSERVATIONS)
    prev_bgr, W, H = None, 0, 0
    for k, (vi, fpth, gs) in enumerate(sampled):
        progress(f"Finding and tracking plants: frame {k + 1}/{len(sampled)}", 0.05 + 0.3 * k / len(sampled))
        if gs < BLUR_FRACTION * median_sharp:                # too blurry to detect reliably
            continue
        bgr = cv2.imread(str(fpth))
        H, W = bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        mask = fp.predict_plant_mask(pil, M["plant_seg"], M["plant_size"])
        dets = []
        for p in fp.crop_masks_by_area(pil, mask):
            if p["mask_area"] < MIN_PLANT_AREA_FRAC * W * H:
                continue
            bx1, by1, bx2, by2 = p["box"]
            dets.append(dict(box=(bx1, by1, bx2, by2), area=p["mask_area"], sharp=sharpness(bgr[by1:by2, bx1:bx2]),
                             edge=bx1 <= 0.01 * W or by1 <= 0.01 * H or bx2 >= 0.99 * W or by2 >= 0.99 * H,
                             emb=_embed(M["dino"], p["crop"])))
        A = estimate_motion(prev_bgr, bgr) if prev_bgr is not None else None
        tracker.update(k, (W, H), dets, A)
        prev_bgr = bgr
    tracks = tracker.finalize()
    progress(f"{len(tracks)} unique plant(s) found", 0.36)

    plants, frame_area = [], W * H
    font = _font(max(16, W // 50))
    for n, t in enumerate(tracks, 1):
        progress(f"Plant {n}/{len(tracks)}: best view, degrade, SR3, lesions, DINOv2", 0.36 + 0.62 * (n - 1) / len(tracks))
        o = pick_frames(t, frame_area, k=1, min_gap=1)[0]     # sharpest / largest / not clipped by the frame edge
        full = Image.open(sampled[o["frame"]][1]).convert("RGB")
        x1, y1, x2, y2 = (int(v) for v in o["box"])
        crop = full.crop((max(0, x1), max(0, y1), min(full.width, x2), min(full.height, y2)))
        res = fp.run_pipeline(crop, sampler=cfg["sampler"], steps=int(cfg["steps"]), det_thresh=cfg["det_thresh"],
                              healthy_pct=cfg["healthy_pct"], single_plant=True)
        if not res["plants"]:
            continue
        pl = build_plant(res["plants"][0], res, crop, crop.size, write, n)
        shown = full.copy()
        d = ImageDraw.Draw(shown)
        d.rectangle((x1, y1, x2, y2), outline=(0, 120, 255), width=max(3, full.width // 300))
        d.text((x1 + 8, y1 + 8), f"Plant {n}", fill=(255, 255, 0), font=font)
        pl.update(frame_image=write(shown, f"plant_{n:02d}/frame"), time_s=round(sampled[o["frame"]][0] / native_fps, 2),
                  frames_seen=len(t.obs))
        plants.append(pl)
    shutil.rmtree(frames_dir, ignore_errors=True)
    return finish(plants, "video", cfg, time.time() - t0,
                  dict(video=dict(fps=native_fps, sampled_frames=len(sampled), tracks=len(tracks))))
