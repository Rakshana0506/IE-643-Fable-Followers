"""
Video pipeline: find every UNIQUE plant in a video, then run the plant pipeline on it.

  video
   └─ sample frames (SAMPLE_FPS), drop blurry ones
       └─ per frame: DeepLab plant mask -> plant crops (+ DINOv2 appearance embedding)
           └─ track plants across frames (camera-motion compensated)  -> one id per physical plant
               └─ per plant: pick the best BEST_FRAMES views (sharp, large, not cut by the frame edge)
                   └─ run final_pipeline.run_pipeline on each view (leaves -> degrade -> SR3 -> lesions -> DINOv2)
                       └─ pool all views of the plant: majority vote for plant + disease, median severity
                           and, per leaf, what survives degrade + SR3 (lesions kept / shifted / lost / new)

Set the variables below, then run:  python video_pipeline.py
Needs final_pipeline.py and plant_tracker.py next to this file, plus opencv-python.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

import final_pipeline as fp
from plant_tracker import PlantTracker, estimate_motion, pick_frames, sharpness
from lesion_compare import compare_lesions, agreement_overlay, leaf_figure, show_unmasked

# ==================================================
# CONFIGURATION
# ==================================================
VIDEO_PATH = r"C:\Users\amals\Downloads\Video Project 13.mp4"
OUT_DIR = Path("video_results")

SAMPLE_FPS = 3.0          # frames analysed per second of video. Raise it if the camera moves fast:
                          # a plant should be visible in ~6+ sampled frames (the script warns when it is not)
BLUR_FRACTION = 0.5       # skip frames sharper-than-median * this (motion blur)
MIN_PLANT_AREA_FRAC = 0.003   # plant mask blobs smaller than this fraction of the frame are ignored
PLANT_AREA_RATIO = 0.05   # keep blobs >= this * the largest blob of the frame (lower than the 0.25 used for photos)

MIN_OBSERVATIONS = 3      # a plant must be seen in this many sampled frames, otherwise it is treated as flicker
MAX_MISSED = 4            # frames a plant may disappear (occlusion, blur) before its track is closed
IOU_MIN = 0.15            # minimum box overlap (after camera-motion compensation) to continue a track

BEST_FRAMES = 3           # views of each plant that go through the full pipeline (SR3 is slow: cost scales with this)
MIN_FRAME_GAP = 2         # views must be at least this many sampled frames apart

DEGRADE_MODE = "sr3_ft"   # the normal 256x256 crop is DEGRADED, then SR3 reconstructs it:
                          #   "sr3_ft" the random degradation SR3 was trained with | "simple" blur+noise+JPEG | "none"
MASK_MIN_BG_FRAC = 0.30       # remove background only when it covers at least this fraction of the crop (0 = always)
FILL_MASK_HOLES = True        # fill holes in the plant mask so lesions inside the leaf are not blacked out
MASK_LEAF_BACKGROUND = True   # leaf crops keep only plant-mask pixels, everything else black (DINOv2 still sees the unmasked crop)
DEGRADE_SEED = 0          # reproducible degradations (leaf n uses seed DEGRADE_SEED + n)
SAMPLER = "ddim_stochastic"          # "ancestral" | "ddim" | "ddim_stochastic"
STEPS = 50                # SR3 steps for the DDIM samplers
SAVE_PREVIEW_VIDEO = True # tracks_preview.mp4: boxes + plant ids on the sampled frames (check identities here!)

OUT_DIR.mkdir(parents=True, exist_ok=True)
frames_dir = OUT_DIR / "frames"
frames_dir.mkdir(exist_ok=True)

# ==================================================
# 1. READ + SAMPLE THE VIDEO
# ==================================================
cap = cv2.VideoCapture(str(VIDEO_PATH))
assert cap.isOpened(), f"Cannot open {VIDEO_PATH}"
native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
every = max(1, round(native_fps / SAMPLE_FPS))
print(f"Video: {native_fps:.1f} fps, sampling every {every} frames (~{native_fps / every:.1f} per second)")

sampled = []                                   # (video_frame_index, path, global sharpness)
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % every == 0:
        path = frames_dir / f"{len(sampled):05d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        sampled.append((i, path, sharpness(small)))
    i += 1
cap.release()
print(f"Sampled {len(sampled)} frames")
assert sampled, "No frames were read"
median_sharp = float(np.median([s[2] for s in sampled]))

# ==================================================
# 2. DETECT PLANTS IN EVERY FRAME AND TRACK THEM
# ==================================================
M = fp.get_models()
dino = M["dino"]


@torch.no_grad()
def embed(img):
    """Appearance embedding of a plant crop: DINOv2 pooled features (L2-normalised)."""
    x = TF.normalize(TF.to_tensor(img.resize((fp.DINO_IMG_SIZE, fp.DINO_IMG_SIZE))),
                     fp.IMAGENET_MEAN, fp.IMAGENET_STD).unsqueeze(0).to(fp.device)
    f = dino.forward_head(dino.forward_features(x), pre_logits=True)[0].float()
    return F.normalize(f, dim=0).cpu().numpy()


fp.AREA_RATIO = PLANT_AREA_RATIO
fp.DEGRADE_MODE, fp.DEGRADE_SEED = DEGRADE_MODE, DEGRADE_SEED
fp.MASK_LEAF_BACKGROUND = MASK_LEAF_BACKGROUND
fp.MASK_MIN_BG_FRAC, fp.FILL_MASK_HOLES = MASK_MIN_BG_FRAC, FILL_MASK_HOLES
tracker = PlantTracker(iou_min=IOU_MIN, max_missed=MAX_MISSED, min_obs=MIN_OBSERVATIONS)
prev_bgr, shifts, used = None, [], 0

for k, (vi, path, gs) in enumerate(sampled):
    if gs < BLUR_FRACTION * median_sharp:          # too blurry to detect or classify reliably
        continue
    bgr = cv2.imread(str(path))
    H, W = bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    mask = fp.predict_plant_mask(pil, M["plant_seg"], M["plant_size"])
    dets = []
    for p in fp.crop_masks_by_area(pil, mask):
        if p["mask_area"] < MIN_PLANT_AREA_FRAC * W * H:
            continue
        x1, y1, x2, y2 = p["box"]
        dets.append(dict(box=(x1, y1, x2, y2), area=p["mask_area"], sharp=sharpness(bgr[y1:y2, x1:x2]),
                         edge=x1 <= 0.01 * W or y1 <= 0.01 * H or x2 >= 0.99 * W or y2 >= 0.99 * H,
                         emb=embed(p["crop"])))

    A = estimate_motion(prev_bgr, bgr) if prev_bgr is not None else None
    if A is not None:
        shifts.append(float(np.hypot(A[0, 2], A[1, 2]) / W))
    tracker.update(k, (W, H), dets, A)
    prev_bgr = bgr
    used += 1
print(f"Analysed {used} sharp frames, {len(tracker.tracks)} tracks created")
frame_area = W * H

if shifts and np.median(shifts) > 0.2:
    print(f"[warn] the camera moves {np.median(shifts) * 100:.0f}% of the frame width between sampled frames. "
          f"Plants are seen in too few frames and may be split or dropped. Raise SAMPLE_FPS to about "
          f"{SAMPLE_FPS * np.median(shifts) / 0.1:.0f}.")

tracks = tracker.finalize()
print(f"Unique plants (seen in >= {MIN_OBSERVATIONS} frames): {len(tracks)}")

# ==================================================
# 3. RUN THE PLANT PIPELINE ON THE BEST VIEWS OF EACH UNIQUE PLANT
# ==================================================
med = lambda xs: float(np.median(xs))
summary = []

for t in tracks:
    tdir = OUT_DIR / f"plant_{t.id:03d}"
    tdir.mkdir(exist_ok=True)
    views = pick_frames(t, frame_area, k=BEST_FRAMES, min_gap=MIN_FRAME_GAP)
    per_view, leaves = [], []
    for o in views:
        full = Image.open(sampled[o["frame"]][1]).convert("RGB")
        x1, y1, x2, y2 = (int(v) for v in o["box"])
        crop = full.crop((max(0, x1), max(0, y1), min(full.width, x2), min(full.height, y2)))
        res = fp.run_pipeline(crop, sampler=SAMPLER, steps=STEPS, single_plant=True)
        if not res["plants"]:
            continue

        # ---- everything for this view goes in its own folder (no mask images: the mask is applied internally) ----
        vdir = tdir / f"frame_{o['frame']:05d}"
        (vdir / "leaves").mkdir(parents=True, exist_ok=True)
        full.save(vdir / "00_original_frame.jpg", quality=95)                 # the video frame as sampled
        crop.save(vdir / "01_plant_crop.jpg", quality=95)                      # this plant, cut from the frame
        cv = res["total"]["canvas"]                                            # lesions of every leaf pasted on the plant crop
        res["lesions_normal_img"].save(vdir / "02_plant_lesions_original.jpg")
        res["lesions_bicubic_img"].save(vdir / "03_plant_lesions_bicubic.jpg")
        res["lesions_sr3_img"].save(vdir / "04_plant_lesions_sr3.jpg")
        # whole-plant agreement with the original: yellow = both, blue = only on original, red = only on bicubic / SR3
        agreement_overlay(crop.convert("RGB"), cv["n"], cv["d"]).save(vdir / "05_plant_bicubic_vs_original.jpg")
        agreement_overlay(crop.convert("RGB"), cv["n"], cv["s"]).save(vdir / "06_plant_sr3_vs_original.jpg")

        leaf_rows = []
        for l in res["leaves"]:
            n = f"leaf_{l['idx']:02d}"
            ld = vdir / "leaves"
            orig = l["img_raw"]                                                  # original leaf crop (256x256)
            m = l["mask256"]
            low = show_unmasked(l["img_low"], l["img_low_full"], m)              # background put back for display only
            bic = show_unmasked(l["img_d"], l["img_d_full"], m)                  # (degraded the same way as the leaf)
            srx = show_unmasked(l["img_s"], orig, m)
            ov_o = fp.overlay_mask(orig, l["lesion_n"])
            ov_b = fp.overlay_mask(bic, l["lesion_d"])
            ov_s = fp.overlay_mask(srx, l["lesion_s"])
            st, mn, ms = compare_lesions(l["lesion_n"], l["lesion_s"])           # SR3 vs original
            st_d, mn2, md = compare_lesions(l["lesion_n"], l["lesion_d"])        # bicubic vs original
            cmp_b = agreement_overlay(orig, mn2, md)
            cmp_s = agreement_overlay(orig, mn, ms)
            orig.save(ld / f"{n}_1_original.png")
            low.save(ld / f"{n}_2_degraded.png")                                 # low-res, as degraded
            bic.save(ld / f"{n}_3_bicubic.png")                                  # degraded, bicubic up-sampled (SR3 input)
            srx.save(ld / f"{n}_4_sr3.png")
            ov_o.save(ld / f"{n}_5_lesions_original.png")
            ov_b.save(ld / f"{n}_6_lesions_bicubic.png")
            ov_s.save(ld / f"{n}_7_lesions_sr3.png")
            cmp_b.save(ld / f"{n}_8_bicubic_vs_original.png")                    # yellow both, blue only original, red only bicubic
            cmp_s.save(ld / f"{n}_9_sr3_vs_original.png")                        # yellow both, blue only original, red only SR3
            leaf_figure(orig, low, bic, srx, ov_o, ov_b, ov_s, cmp_b, cmp_s,
                        f"Plant {t.id}, frame {o['frame']}, leaf {l['idx']}", st_d, st).save(ld / f"{n}_figure.png")
            leaf_rows.append(dict(leaf=l["idx"], sev_normal=round(l["sev_n"], 2), sev_bicubic=round(l["sev_d"], 2),
                                  sev_sr3=round(l["sev_s"], 2), mask_iou=round(l["iou"], 3), mask_iou_bicubic=round(l["iou_d"], 3),
                                  plant=l["plant"], diagnosis=l["diagnosis"], disease_prob=round(l["disease_prob"], 1),
                                  **st, **{f"bic_{k}": v for k, v in st_d.items()}))
        pl = res["plants"][0]
        view_row = dict(frame=o["frame"], video_frame=sampled[o["frame"]][0], n_leaves=pl["n_leaves"],
                        plant_type=pl["plant_type"], diagnosis=pl["diagnosis"], sev_n=pl["sev_n"], sev_d=pl["sev_d"], sev_s=pl["sev_s"],
                        leaves=leaf_rows)
        (vdir / "view.json").write_text(json.dumps(view_row, indent=1))
        per_view.append(view_row)
        leaves += res["leaves"]
    if not per_view:
        continue

    all_leaf_rows = [r for v in per_view for r in v["leaves"]]
    tot = lambda key: int(sum(r[key] for r in all_leaf_rows))

    # pool EVERY leaf of EVERY view: majority vote for the plant, then the diagnosis among that plant's leaves
    plant, pn = fp.majority([(l["plant"], l["plant_conf"]) for l in leaves])
    sel = [l for l in leaves if l["plant"] == plant]
    diag, dn = fp.majority([(l["diagnosis"], l["top_conf"]) for l in sel])
    agree_l = [l for l in sel if l["diagnosis"] == diag]
    row = dict(
        plant_id=t.id,
        first_time_s=round(sampled[t.obs[0]["frame"]][0] / native_fps, 2),
        last_time_s=round(sampled[t.obs[-1]["frame"]][0] / native_fps, 2),
        frames_seen=len(t.obs), views_analysed=len(per_view), leaves_pooled=len(leaves),
        plant_type=plant, plant_votes=f"{pn}/{len(leaves)}",
        diagnosis=diag, diagnosis_votes=f"{dn}/{len(sel)}",
        disease_prob=round(float(np.mean([l["disease_prob"] for l in agree_l])), 1),
        severity_normal=round(med([v["sev_n"] for v in per_view]), 2),
        severity_bicubic=round(med([v["sev_d"] for v in per_view]), 2),
        severity_sr3=round(med([v["sev_s"] for v in per_view]), 2),
        severity_range_normal=f"{min(v['sev_n'] for v in per_view):.1f}-{max(v['sev_n'] for v in per_view):.1f}",
        # how much SR3 changes what the lesion model sees (summed over every leaf of every view)
        lesions_on_normal=tot("lesions_normal"), lesions_on_sr3=tot("lesions_sr3"), lesions_kept=tot("kept"),
        lesions_shifted=tot("shifted"), lesions_lost_in_sr3=tot("lost_in_sr3"), lesions_new_in_sr3=tot("new_in_sr3"),
        # the same for plain bicubic (degraded input, no SR3): what the lesion model finds without super-resolution
        lesions_on_bicubic=tot("bic_lesions_sr3"), bic_kept=tot("bic_kept"), bic_shifted=tot("bic_shifted"),
        bic_lost=tot("bic_lost_in_sr3"), bic_new=tot("bic_new_in_sr3"),
        mean_iou_bicubic=round(float(np.mean([r["mask_iou_bicubic"] for r in all_leaf_rows])), 3),
        mean_iou_sr3=round(float(np.mean([r["mask_iou"] for r in all_leaf_rows])), 3),
        sev_err_bicubic=round(float(np.mean([abs(r["sev_bicubic"] - r["sev_normal"]) for r in all_leaf_rows])), 2),
        sev_err_sr3=round(float(np.mean([abs(r["sev_sr3"] - r["sev_normal"]) for r in all_leaf_rows])), 2),
    )
    summary.append(row)
    (tdir / "summary.json").write_text(json.dumps(dict(row, views=per_view), indent=1))
    print(f"Plant {t.id}: {plant} ({row['plant_votes']}), {diag} ({row['diagnosis_votes']}), "
          f"severity {row['severity_normal']}% normal / {row['severity_bicubic']}% bicubic / {row['severity_sr3']}% SR3, "
          f"lesions SR3 kept {row['lesions_kept']} lost {row['lesions_lost_in_sr3']} new {row['lesions_new_in_sr3']} | "
          f"bicubic kept {row['bic_kept']} lost {row['bic_lost']} new {row['bic_new']}, "
          f"{len(per_view)} views, seen {row['first_time_s']}s-{row['last_time_s']}s")

if summary:
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))

    # ---- report: normal crop = reference; does SR3 beat plain bicubic on the same degraded input? ----
    S = lambda k: sum(r[k] for r in summary)
    M = lambda k: float(np.mean([r[k] for r in summary]))
    ref = S("lesions_on_normal")
    def pct(a): return f"{100 * a / ref:.0f}%" if ref else "n/a"
    lines = [
        "# Degraded-input comparison: bicubic vs SR3 (reference = lesions on the normal crop)", "",
        f"Plants: {len(summary)} | degradation: {DEGRADE_MODE} | sampler: {SAMPLER}, {STEPS} steps | "
        f"background mask: {'on' if MASK_LEAF_BACKGROUND else 'off'}", "",
        "| metric | bicubic (no SR) | SR3 |", "|---|---|---|",
        f"| lesions found (reference has {ref}) | {S('lesions_on_bicubic')} | {S('lesions_on_sr3')} |",
        f"| kept (recall) | {S('bic_kept')} ({pct(S('bic_kept'))}) | {S('lesions_kept')} ({pct(S('lesions_kept'))}) |",
        f"| lost | {S('bic_lost')} | {S('lesions_lost_in_sr3')} |",
        f"| new / spurious | {S('bic_new')} | {S('lesions_new_in_sr3')} |",
        f"| shifted | {S('bic_shifted')} | {S('lesions_shifted')} |",
        f"| mean lesion-mask IoU vs normal | {M('mean_iou_bicubic'):.3f} | {M('mean_iou_sr3'):.3f} |",
        f"| mean |severity error| vs normal (pct points) | {M('sev_err_bicubic'):.2f} | {M('sev_err_sr3'):.2f} |", "",
    ]
    better = [("recall", S("lesions_kept") > S("bic_kept")), ("fewer spurious lesions", S("lesions_new_in_sr3") < S("bic_new")),
              ("higher IoU", M("mean_iou_sr3") > M("mean_iou_bicubic")), ("lower severity error", M("sev_err_sr3") < M("sev_err_bicubic"))]
    win = [n for n, ok in better if ok]
    lines.append("SR3 beats bicubic on: " + (", ".join(win) if win else "nothing; plain bicubic is as good or better here") + ".")
    lines.append("Lost / new / kept are relative to the normal crop, not ground truth.")
    (OUT_DIR / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))

# ==================================================
# 4. PREVIEW VIDEO: check that every physical plant keeps ONE id
# ==================================================
if SAVE_PREVIEW_VIDEO and tracks:
    by_frame = {}
    for t in tracks:
        for o in t.obs:
            by_frame.setdefault(o["frame"], []).append((t.id, o["box"]))
    first = cv2.imread(str(sampled[0][1]))
    scale = min(1.0, 1280 / first.shape[1])
    size = (int(first.shape[1] * scale), int(first.shape[0] * scale))
    out = cv2.VideoWriter(str(OUT_DIR / "tracks_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), SAMPLE_FPS, size)
    rng = np.random.default_rng(1)
    colors = {t.id: tuple(int(c) for c in rng.integers(60, 255, 3)) for t in tracks}
    for k, (vi, path, gs) in enumerate(sampled):
        img = cv2.resize(cv2.imread(str(path)), size)
        for tid, (x1, y1, x2, y2) in by_frame.get(k, []):
            p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
            cv2.rectangle(img, p1, p2, colors[tid], 3)
            cv2.putText(img, f"plant {tid}", (p1[0] + 6, p1[1] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[tid], 2)
        out.write(img)
    out.release()
    print("Wrote", OUT_DIR / "tracks_preview.mp4")

print("Done. Results in", OUT_DIR.resolve())
