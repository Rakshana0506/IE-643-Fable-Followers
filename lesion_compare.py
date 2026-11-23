"""
Lesion comparison between the NORMAL (degraded input) leaf crop and its SR3 reconstruction, plus the
per-leaf figure (numpy, scipy, PIL only).

The normal crop is the reference (the "original"); SR3 is run on a DEGRADED copy of it, so this measures what
survives degrade + reconstruct, relative to the lesions the lesion model finds on the normal crop:
  kept         lesion found on both
  shifted      kept, but its centre moved by more than `shift_px` pixels
  lost_in_sr3  lesion on the normal crop that is gone after degrade + SR3 (disappeared)
  new_in_sr3   lesion that only appears after degrade + SR3 (spurious / invented detail)
(The normal crop itself is real footage, not ground truth, so these are agreement counts, not accuracy.)
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

YELLOW, BLUE, RED = (255, 215, 0), (40, 120, 255), (230, 50, 40)


def _clean(mask, min_area):
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, bool), lab, 0
    areas = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, a in enumerate(areas) if a >= min_area])
    lab2, n2 = ndimage.label(keep)
    return keep, lab2, n2


def compare_lesions(les_n, les_s, min_area=15, cover=0.3, shift_px=8):
    """les_n / les_s: bool masks on the same 256x256 grid. Specks smaller than `min_area` px are ignored."""
    mn, ln, nn = _clean(les_n, min_area)
    ms, ls, ns = _clean(les_s, min_area)
    kept = shifted = lost = new = 0
    shifts = []
    for i in range(1, nn + 1):
        comp = ln == i
        if (comp & ms).sum() / comp.sum() >= cover:
            touching = np.unique(ls[comp & ms])
            touching = touching[touching > 0]
            other = np.isin(ls, touching)
            d = float(np.hypot(*(np.array(ndimage.center_of_mass(comp)) - np.array(ndimage.center_of_mass(other)))))
            kept += 1
            shifts.append(d)
            shifted += d > shift_px
        else:
            lost += 1
    for j in range(1, ns + 1):
        comp = ls == j
        if (comp & mn).sum() / comp.sum() < cover:
            new += 1
    stats = dict(lesions_normal=nn, lesions_sr3=ns, kept=kept, shifted=int(shifted), lost_in_sr3=lost, new_in_sr3=new,
                 median_shift_px=float(np.median(shifts)) if shifts else 0.0,
                 px_both=int((mn & ms).sum()), px_only_normal=int((mn & ~ms).sum()), px_only_sr3=int((ms & ~mn).sum()))
    return stats, mn, ms


def agreement_overlay(base, mn, ms, alpha=0.65):
    """base image with lesions coloured: both = yellow, only on the normal crop = blue, only after SR3 = red."""
    arr = np.array(base.convert("RGB")).astype(np.float32)
    for m, c in ((mn & ms, YELLOW), (mn & ~ms, BLUE), (ms & ~mn, RED)):
        arr[m] = (1 - alpha) * arr[m] + alpha * np.array(c, np.float32)
    return Image.fromarray(arr.astype(np.uint8))


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def show_unmasked(img, raw, mask256):
    """
    The background mask is applied INSIDE the pipeline (black background fed to bicubic / SR3 / lesion model).
    For the saved results the original background is put back outside the mask, so no mask or black area is visible.
    """
    if mask256 is None:
        return img
    w, h = img.size
    m = np.array(Image.fromarray(mask256.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
    out = np.array(img.convert("RGB")).copy()
    bg = np.array(raw.convert("RGB").resize((w, h), Image.LANCZOS))
    out[~m] = bg[~m]
    return Image.fromarray(out)


def leaf_figure(orig, degraded, bicubic, sr3, ov_o, ov_b, ov_s, cmp_b, cmp_s, title, st_b, st_s):
    """
    Row 1: original | degraded (low-res, shown pixelated) | bicubic | SR3
    Row 2: lesions on original | lesions on bicubic | lesions on SR3 | bicubic vs original | SR3 vs original
    """
    row1 = [("Original", orig), ("Degraded (low-res)", degraded), ("Bicubic", bicubic), ("SR3", sr3)]
    row2 = [("Lesions: original", ov_o), ("Lesions: bicubic", ov_b), ("Lesions: SR3", ov_s),
            ("Bicubic vs original", cmp_b), ("SR3 vs original", cmp_s)]
    s = orig.size[0]
    head, label, foot, pad = 54, 20, 70, 6
    W = 5 * s + 6 * pad
    H = head + 2 * (label + s + pad) + foot
    fig = Image.new("RGB", (W, H), (248, 250, 249))
    d = ImageDraw.Draw(fig)
    d.text((pad + 2, 6), title, fill=(20, 35, 29), font=_font(17))
    for r, row in enumerate((row1, row2)):
        y = head + r * (label + s + pad)
        for i, (name, im) in enumerate(row):
            x = pad + i * (s + pad)
            d.text((x + 2, y), name, fill=(70, 85, 78), font=_font(13))
            fig.paste(im.convert("RGB").resize((s, s), Image.NEAREST if name.startswith("Degraded") else Image.BICUBIC),
                      (x, y + label))
    y = head + 2 * (label + s + pad) + 4
    for k, (name, st) in enumerate((("Bicubic", st_b), ("SR3", st_s))):
        d.text((pad + 2, y + k * 18),
               f"{name}: {st['lesions_sr3']} lesions vs {st['lesions_normal']} on original.  Kept {st['kept']} "
               f"(shifted {st['shifted']}), lost {st['lost_in_sr3']}, new {st['new_in_sr3']}.",
               fill=(20, 35, 29), font=_font(13))
    x, y = pad + 2, y + 40
    for c, label_ in ((YELLOW, "agreement"), (BLUE, "lesion on original only (missed)"), (RED, "not on original (spurious)")):
        d.rectangle((x, y + 3, x + 10, y + 13), fill=c)
        d.text((x + 15, y), label_, fill=(70, 85, 78), font=_font(13))
        x += 40 + int(d.textlength(label_, font=_font(13)))
    return fig
