"""Quiver web server: upload a clip, get the diagnosis, the sound, the magnified video and a work order."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from quiver import llm, pipeline, reference
from quiver.diagnose import resolve_alias

ROOT = Path(__file__).resolve().parent
RUNS = Path(os.getenv("QUIVER_RUNS", ROOT / "runs"))
RUNS.mkdir(parents=True, exist_ok=True)
MAX_BYTES = 300 * 1024 * 1024

app = FastAPI(title="Quiver")


def _num(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        raise HTTPException(400, f"not a number: {v}")


@app.get("/api/health")
def health():
    return {"ok": True, "llm": llm.configured(), "model": os.getenv("QUIVER_LLM_MODEL", "rules")}


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...), asset: str = Form("Machine"), rpm: str | None = Form(None),
                  mm_per_px: str | None = Form(None), fps: str | None = Form(None),
                  baseline_id: str | None = Form(None), parent_id: str | None = Form(None)):
    run_id = uuid.uuid4().hex[:10]
    d = RUNS / run_id
    d.mkdir()
    src = d / ("input" + (Path(video.filename or "").suffix or ".mp4"))
    size = 0
    with src.open("wb") as f:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, "clip too large (300 MB max)")
            f.write(chunk)

    baseline_1x = None
    if baseline_id:
        b = _load(baseline_id)
        baseline_1x = b["diagnosis"]["orders"].get("1x")

    try:
        result = await run_in_threadpool(pipeline.run, str(src), str(d), _num(rpm), _num(mm_per_px), baseline_1x, _num(fps))
    except ValueError as e:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(422, str(e))

    # Second recording of the same machine at a different frame rate: settle aliasing.
    if parent_id:
        p = _load(parent_id)
        pa, ra = p["diagnosis"]["peaks"], result["diagnosis"]["peaks"]
        if pa and ra:
            fa, fb = p["diagnosis"]["quality"]["fps"], result["diagnosis"]["quality"]["fps"]
            result["alias_check"] = {
                "first": {"fps": fa, "peak_hz": pa[0]["hz"]},
                "second": {"fps": fb, "peak_hz": ra[0]["hz"]},
                "true_hz_candidates": resolve_alias(pa[0]["hz"], fa, ra[0]["hz"], fb, f_max=max(fa, fb) * 2)[:5],
            }

    result["work_order"] = await run_in_threadpool(llm.write_work_order, result, str(d), asset)
    result["id"] = run_id
    result["asset"] = asset
    (d / "result.json").write_text(json.dumps(result))
    return result


@app.post("/api/runs/{run_id}/reference")
async def add_reference(run_id: str, csv: UploadFile = File(...)):
    """Ground truth: a phyphox accelerometer export recorded on the machine at the same time."""
    result = _load(run_id)
    data = await csv.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "CSV too large")
    try:
        t, acc = reference.parse_phyphox(data)
        ref = reference.spectrum(t, acc, max_hz=result["diagnosis"]["quality"]["nyquist_hz"])
    except (ValueError, IndexError) as e:
        raise HTTPException(422, f"could not read that CSV: {e}")
    result["reference"] = ref
    result["reference_check"] = reference.compare(ref, result["diagnosis"]["peaks"])
    (RUNS / run_id / "result.json").write_text(json.dumps(result))
    return result


def _load(run_id: str) -> dict:
    if not run_id.isalnum():
        raise HTTPException(400, "bad id")
    f = RUNS / run_id / "result.json"
    if not f.exists():
        raise HTTPException(404, "unknown run")
    return json.loads(f.read_text())


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return _load(run_id)


@app.get("/runs/{run_id}/{name}")
def run_file(run_id: str, name: str):
    if not run_id.isalnum() or name not in {"true.wav", "audible.wav", "heatmap.png", "frame.jpg", "magnified.mp4"}:
        raise HTTPException(404)
    f = RUNS / run_id / name
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
