"""
Plant tracking across video frames (no deep-learning dependencies: numpy, scipy, OpenCV).

Plants stay still in the world while the camera moves, so each new frame is first aligned to the previous
one (feature tracking + a similarity transform). Every known plant box is warped by that transform and then
matched to the plant masks found in the new frame by box overlap (IoU) plus appearance similarity.
A plant that is matched keeps its id; an unmatched detection starts a new plant.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------- geometry
def box_iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0, inter / max(min(area_a, area_b), 1e-9)


def warp_box(box, A):
    """Warp an (x1, y1, x2, y2) box with a 2x3 affine matrix; returns the bounding box of the warped corners."""
    if A is None:
        return tuple(box)
    x1, y1, x2, y2 = box
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    out = pts @ A[:, :2].T + A[:, 2]
    return (float(out[:, 0].min()), float(out[:, 1].min()), float(out[:, 0].max()), float(out[:, 1].max()))


def estimate_motion(prev_bgr, cur_bgr, work_side=480, min_inliers=15):
    """
    2x3 similarity transform that maps points of the previous frame onto the current frame
    (full-resolution pixel coordinates). Returns None when the frames cannot be aligned.
    """
    h, w = prev_bgr.shape[:2]
    f = min(1.0, work_side / max(h, w))
    prev = cv2.cvtColor(cv2.resize(prev_bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    cur = cv2.cvtColor(cv2.resize(cur_bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev, maxCorners=500, qualityLevel=0.01, minDistance=7)
    if p0 is None or len(p0) < min_inliers:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < min_inliers:
        return None
    A, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is None or inl is None or int(inl.sum()) < min_inliers:
        return None
    A = A.astype(np.float64)
    A[:, 2] /= f                      # translation back to full-resolution pixels
    return A


def sharpness(bgr, side=256):
    """Variance of the Laplacian on a fixed-size grey version (comparable between crops of different size)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = side / max(g.shape)
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# ---------------------------------------------------------------- tracker
class Track:
    def __init__(self, tid, det, frame_i):
        self.id = tid
        self.pred_box = tuple(det["box"])      # where the plant should be in the CURRENT frame
        self.emb = det.get("emb")
        self.missed = 0
        self.active = True
        self.obs = []                          # one record per frame the plant was actually seen
        self.add(det, frame_i)

    def add(self, det, frame_i):
        self.pred_box = tuple(det["box"])
        if det.get("emb") is not None:
            self.emb = det["emb"] if self.emb is None else 0.7 * self.emb + 0.3 * det["emb"]
            self.emb = self.emb / (np.linalg.norm(self.emb) + 1e-9)
        self.missed = 0
        self.obs.append(dict(frame=frame_i, **det))


class PlantTracker:
    def __init__(self, iou_min=0.15, w_app=0.25, max_missed=4, min_obs=3, border_margin=0.25):
        self.iou_min, self.w_app, self.max_missed = iou_min, w_app, max_missed
        self.min_obs, self.border_margin = min_obs, border_margin
        self.tracks, self._next = [], 1

    def update(self, frame_i, frame_size, dets, A=None):
        """
        frame_size = (W, H). dets = [{box, area, sharp, edge, emb?}]. A = transform previous frame -> this frame.
        """
        W, H = frame_size
        active = [t for t in self.tracks if t.active]
        for t in active:                                    # move every known plant into this frame
            t.pred_box = warp_box(t.pred_box, A)

        cost = np.full((len(active), len(dets)), 1e3)
        for i, t in enumerate(active):
            for j, d in enumerate(dets):
                iou, ioa = box_iou(t.pred_box, d["box"])
                overlap = max(iou, 0.7 * ioa)               # IoA copes with plants cut by the frame edge
                if overlap < self.iou_min:
                    continue
                app = float(np.dot(t.emb, d["emb"])) if (t.emb is not None and d.get("emb") is not None) else 1.0
                cost[i, j] = (1 - overlap) * (1 - self.w_app) + (1 - app) * self.w_app
        matched_t, matched_d = set(), set()
        if len(active) and len(dets):
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e3:
                    active[i].add(dets[j], frame_i)
                    matched_t.add(i); matched_d.add(j)

        for i, t in enumerate(active):                      # plants not seen in this frame
            if i in matched_t:
                continue
            t.missed += 1
            cx, cy = (t.pred_box[0] + t.pred_box[2]) / 2, (t.pred_box[1] + t.pred_box[3]) / 2
            gone = not (-self.border_margin * W < cx < (1 + self.border_margin) * W and
                        -self.border_margin * H < cy < (1 + self.border_margin) * H)
            if gone or t.missed > self.max_missed:
                t.active = False

        for j, d in enumerate(dets):                        # new plants
            if j not in matched_d:
                self.tracks.append(Track(self._next, d, frame_i))
                self._next += 1

    def finalize(self):
        """Plants seen in at least `min_obs` frames; shorter tracks are treated as flicker."""
        return [t for t in self.tracks if len(t.obs) >= self.min_obs]


# ---------------------------------------------------------------- choosing the views to analyse
def pick_frames(track, frame_area, k=3, min_gap=2):
    """
    Best k observations of a plant: sharp, large, not cut by the frame edge, and at least `min_gap` sampled
    frames apart so the views differ (motion blur, occlusion and lighting do not repeat in every view).
    """
    sharps = np.array([o["sharp"] for o in track.obs])
    ref = np.median(sharps) + 1e-9
    def score(o):
        size = (o["area"] / frame_area) ** 0.5
        return min(o["sharp"] / ref, 2.0) * size * (0.5 if o["edge"] else 1.0)
    chosen = []
    for o in sorted(track.obs, key=score, reverse=True):
        if all(abs(o["frame"] - c["frame"]) >= min_gap for c in chosen):
            chosen.append(o)
        if len(chosen) == k:
            break
    return sorted(chosen, key=lambda o: o["frame"])
