"""The AI technician: Gemma (served on AMD GPUs) turns measured evidence into a work order.

Any OpenAI-compatible endpoint works. On the AMD Developer Cloud that is the
event's Token Factory, or a vLLM server on an Instinct GPU:

    QUIVER_LLM_BASE_URL=https://.../v1
    QUIVER_LLM_API_KEY=...
    QUIVER_LLM_MODEL=google/gemma-3-27b-it

Without those variables Quiver falls back to a rules-only work order, so the
pipeline still runs offline.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

import httpx

SYSTEM = """You are Quiver, a senior vibration analyst writing for a plant maintenance team.
You receive measurements extracted from a phone video of a running machine: the vibration
spectrum peaks, the amplitude at each multiple of running speed, quality checks, and rule-based
findings. You may also receive an image of the machine with the vibrating regions highlighted.

Rules:
- Every claim must cite a number from the evidence (a frequency, an order amplitude, an SNR).
- Never invent measurements. If the evidence is weak, say what to record next instead of guessing.
- If the agent action is "recapture", do not diagnose; explain what is missing and how to re-film.
- Amplitudes are in pixels of image motion unless a mm/s value is given.
Reply with JSON only:
{"headline": str, "diagnosis": str, "why": [str], "confidence": "low"|"medium"|"high",
 "next_step": str,
 "work_order": {"title": str, "priority": "P1"|"P2"|"P3"|"P4", "asset": str,
                "tasks": [str], "parts": [str], "estimated_downtime_h": number}}"""


def configured() -> bool:
    return bool(os.getenv("QUIVER_LLM_BASE_URL") and os.getenv("QUIVER_LLM_MODEL"))


def _evidence(result: dict, asset: str) -> dict:
    d = result["diagnosis"]
    return {
        "asset": asset,
        "running_speed_hz": d["running_hz"],
        "running_speed_source": d["running_source"],
        "order_amplitudes_px": d["orders"],
        "top_peaks": d["peaks"][:6],
        "quality": d["quality"],
        "rule_findings": d["findings"],
        "severity": d["severity"],
        "agent_action": d["action"],
    }


def write_work_order(result: dict, out_dir: str, asset: str = "unnamed machine") -> dict:
    ev = _evidence(result, asset)
    if not configured():
        return _rules_only(ev)

    content: list[dict] = [{"type": "text", "text": "Evidence:\n" + json.dumps(ev, indent=1)}]
    img = Path(out_dir) / "heatmap.png"
    if img.exists() and os.getenv("QUIVER_LLM_VISION", "1") == "1":
        b64 = base64.b64encode(img.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    body = {
        "model": os.environ["QUIVER_LLM_MODEL"],
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": 900,
    }
    headers = {"Authorization": f"Bearer {os.getenv('QUIVER_LLM_API_KEY', 'none')}"}
    url = os.environ["QUIVER_LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    try:
        r = httpx.post(url, json=body, headers=headers, timeout=90)
        if r.status_code == 400 and len(content) > 1:
            # Text-only deployment: retry without the image.
            body["messages"][1]["content"] = content[:1]
            r = httpx.post(url, json=body, headers=headers, timeout=90)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        out = json.loads(text[text.find("{"): text.rfind("}") + 1])
        out["model"] = os.environ["QUIVER_LLM_MODEL"]
        out["grounding"] = grounding(out, ev)
        if out["grounding"]["unverified"]:  # a number we never measured: don't let it sound sure
            out["confidence"] = "low"
        return out
    except Exception as e:  # keep the demo alive if the endpoint hiccups
        out = _rules_only(ev)
        out["llm_error"] = str(e)[:300]
        return out


NUMBER = re.compile(r"(?<![A-Za-z\d.])(\d+(?:\.\d+)?)(?!\d)(\s*[x×](?![A-Za-z]))?")


def _measured(ev: dict) -> list[float]:
    """Every number in the evidence, plus the few the reply may fairly derive from them:
    a frequency in rpm, and one order amplitude as a percentage of another."""
    nums: list[float] = []

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            nums.append(float(v))
        elif isinstance(v, str):
            nums.extend(float(m.group(1)) for m in NUMBER.finditer(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(ev)
    hz = [ev.get("running_speed_hz")] + [p.get("hz") for p in ev.get("top_peaks", [])]
    nums += [h * 60 for h in hz if h]
    amps = [a for a in (ev.get("order_amplitudes_px") or {}).values() if a]
    nums += [100 * a / b for a in amps for b in amps if a != b]
    return nums


def _matches(n: float, decimals: int, measured: list[float]) -> bool:
    slack = 0.5 * 10 ** -decimals  # the reply may round
    return any(abs(n - m) <= max(slack, 0.02 * abs(m)) for m in measured)


def grounding(reply: dict, ev: dict) -> dict:
    """Check that every number Gemma uses to explain itself was actually measured.
    Order labels like 1x or 2× aren't measurements, and the work order (priority,
    downtime) is planning, so only the headline, diagnosis and reasons are checked."""
    measured = _measured(ev)
    claims = [reply.get("headline"), reply.get("diagnosis"), *(reply.get("why") or [])]
    checked, unverified = 0, []
    for claim in claims:
        if not isinstance(claim, str):
            continue
        for m in NUMBER.finditer(claim):
            if m.group(2):  # an order label: 1x, 2×
                continue
            raw = m.group(1)
            checked += 1
            if not _matches(float(raw), len(raw.split(".")[1]) if "." in raw else 0, measured):
                unverified.append({"number": raw, "in": claim})
    return {"checked": checked, "unverified": unverified}


def _rules_only(ev: dict) -> dict:
    action = ev["agent_action"]
    findings = ev["rule_findings"]
    if action["kind"] == "recapture" or not findings:
        return {"headline": "Need a better recording", "diagnosis": action["reason"], "why": action["instructions"],
                "confidence": "low", "next_step": (action["instructions"] or ["Re-record"])[0],
                "work_order": None, "model": "rules"}
    top = findings[0]
    pri = "P1" if action["kind"] == "escalate" else "P2" if top["confidence"] > 0.7 else "P3"
    tasks = {
        "mass imbalance": ["Lock out and tag out", "Inspect rotor/blades for build-up, damage or missing balance weights",
                           "Field-balance the rotor", "Re-record with Quiver to confirm 1x dropped"],
        "misalignment": ["Lock out and tag out", "Check coupling alignment with a laser tool", "Correct soft foot",
                         "Re-record with Quiver to confirm 2x dropped"],
        "mechanical looseness": ["Torque-check mounting bolts and base plate", "Inspect bearing housing fit",
                                 "Re-record with Quiver"],
    }.get(top["fault"], ["Inspect the highlighted area", "Re-record with Quiver"])
    return {
        "headline": f"Likely {top['fault']}",
        "diagnosis": f"{top['fault']} (rule confidence {top['confidence']:.0%})",
        "why": top["evidence"],
        "confidence": "medium" if top["confidence"] > 0.6 else "low",
        "next_step": tasks[0],
        "work_order": {"title": f"{top['fault'].capitalize()} on {ev['asset']}", "priority": pri, "asset": ev["asset"],
                       "tasks": tasks, "parts": [], "estimated_downtime_h": 1.5},
        "model": "rules",
    }
