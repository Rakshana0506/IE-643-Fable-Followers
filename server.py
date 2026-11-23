"""
Backend for index.html: upload an IMAGE or a VIDEO, get the full analysis per plant.

    pip install flask opencv-python
    python server.py          # then open http://127.0.0.1:5000

Keep this next to index.html, analysis.py, final_pipeline.py, lesion_compare.py, plant_tracker.py,
the sr3_files folder and the checkpoints. Results are written to ./web_results/<job id>/ (the newest 20 are kept).

A run takes minutes (SR3 sampling), so /api/run starts a background job and the page polls /api/jobs/<id>.
"""
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory
from flask.json.provider import DefaultJSONProvider
from PIL import Image

import analysis
import final_pipeline as core


class NumpyJSON(DefaultJSONProvider):
    """Pipeline results can hold numpy numbers; plain Flask JSON rejects them."""
    @staticmethod
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return DefaultJSONProvider.default(o)


HERE = Path(__file__).parent
RESULTS = HERE / "web_results"
RESULTS.mkdir(exist_ok=True)
KEEP_JOBS = 20
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

app = Flask(__name__, static_folder=None)
app.json = NumpyJSON(app)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3        # 2 GB uploads
jobs = {}                                                # id -> state
jobs_lock = threading.Lock()
gpu_lock = threading.Lock()                              # one pipeline run at a time


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.get("/api/config")
def config():
    d = analysis.DEFAULTS
    return jsonify(
        samplers=[{"id": k, "label": v} for k, v in core.SAMPLERS.items()],
        degrade_modes=[{"id": "sr3_ft", "label": "SR3 training degradation (random blur, noise, compression)"},
                       {"id": "simple", "label": "Simple (blur + noise + JPEG)"},
                       {"id": "none", "label": "Plain down-scale only"}],
        defaults=dict(d, healthy_pct=core.HEALTHY_PCT, det_thresh=core.DETECTION_THRESHOLD),
    )


@app.post("/api/run")
def run():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="No file uploaded."), 400
    d = analysis.DEFAULTS
    try:
        cfg = dict(
            sampler=request.form.get("sampler", d["sampler"]),
            steps=max(1, min(1000, int(float(request.form.get("steps", d["steps"]))))),
            degrade_mode=request.form.get("degrade_mode", d["degrade_mode"]),
            mask_background=request.form.get("mask_background", "1") in ("1", "true", "on"),
            mask_min_bg_frac=max(0.0, min(1.0, float(request.form.get("mask_min_bg_pct", 30)) / 100)),
            fill_holes=True,                                 # always on; only matters when the background is masked
            healthy_pct=core.HEALTHY_PCT,
            det_thresh=core.DETECTION_THRESHOLD,
            sample_fps=max(0.2, min(30.0, float(request.form.get("sample_fps", d["sample_fps"])))),
        )
    except ValueError:
        return jsonify(error="Invalid numeric setting."), 400
    if cfg["sampler"] not in core.SAMPLERS:
        return jsonify(error=f"Unknown sampler '{cfg['sampler']}'."), 400
    if cfg["degrade_mode"] not in ("sr3_ft", "simple"):
        return jsonify(error="Unknown degradation mode."), 400
    if any(j["status"] == "running" for j in jobs.values()):
        return jsonify(error="Another analysis is still running. Wait for it to finish."), 409

    job_id = uuid.uuid4().hex[:10]
    out_dir = RESULTS / job_id
    out_dir.mkdir(parents=True)
    is_video = Path(f.filename).suffix.lower() in VIDEO_EXT or (f.mimetype or "").startswith("video/")
    src = out_dir / ("_upload" + (Path(f.filename).suffix.lower() or ".bin"))
    f.save(src)
    job = dict(id=job_id, status="running", progress=[], frac=0.0, started=time.time(), mode="video" if is_video else "image")
    with jobs_lock:
        jobs[job_id] = job
    threading.Thread(target=work, args=(job, src, out_dir, cfg, is_video), daemon=True).start()
    return jsonify(job_id=job_id, mode=job["mode"])


def work(job, src, out_dir, cfg, is_video):
    def progress(msg, frac):
        job["progress"].append(msg)
        job["frac"] = max(job["frac"], float(frac))

    try:
        with gpu_lock:
            url_base = f"/files/{job['id']}"
            if is_video:
                result = analysis.analyse_video(src, cfg, out_dir, url_base, progress)
            else:
                try:
                    image = Image.open(src).convert("RGB")
                except Exception:
                    raise ValueError("That file could not be read as an image or a video.")
                result = analysis.analyse_image(image, cfg, out_dir, url_base, progress)
        src.unlink(missing_ok=True)
        job["result"], job["status"], job["frac"] = result, "done", 1.0
    except Exception as e:                                   # shown in the page instead of a silent failure
        app.logger.exception("analysis failed")
        job["error"], job["status"] = f"{type(e).__name__}: {e}", "error"
    finally:
        prune()


def prune():
    dirs = sorted((p for p in RESULTS.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in dirs[KEEP_JOBS:]:
        shutil.rmtree(p, ignore_errors=True)
        jobs.pop(p.name, None)


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify(error="Unknown job (the server may have been restarted)."), 404
    out = dict(status=job["status"], frac=job["frac"], progress=job["progress"][-30:],
               elapsed=time.time() - job["started"], mode=job["mode"])
    if job["status"] == "done":
        out["result"] = job["result"]
    if job["status"] == "error":
        out["error"] = job["error"]
    return jsonify(out)


@app.get("/api/jobs/<job_id>/zip")
def job_zip(job_id):
    folder = RESULTS / job_id
    if not folder.is_dir():
        abort(404)
    archive = shutil.make_archive(str(RESULTS / f"{job_id}_results"), "zip", folder)
    return send_from_directory(RESULTS, Path(archive).name, as_attachment=True)


@app.get("/files/<job_id>/<path:rel>")
def files(job_id, rel):
    return send_from_directory(RESULTS / job_id, rel)


if __name__ == "__main__":
    core.get_models()                 # load every checkpoint once, up front
    app.run(host="127.0.0.1", port=5000, threaded=True)
