"""The Gemma step: reply parsing, the text-only retry, the rules fallback, and the check
that every number Gemma explains itself with was actually measured."""

import copy
import json as json_mod

import httpx
import pytest

from quiver import llm

RESULT = {"diagnosis": {
    "running_hz": 19.500969665064154, "running_source": "dominant peak",
    "orders": {"0.5x": 0.0017, "1x": 0.2411, "2x": 0.042, "3x": 0.0041},
    "peaks": [{"hz": 19.5, "amp_px": 0.2411, "snr_db": 48.8}, {"hz": 39.0, "amp_px": 0.042, "snr_db": 33.6}],
    "quality": {"fps": 240.0, "nyquist_hz": 120.0, "duration_s": 4.0},
    "findings": [{"fault": "mass imbalance", "confidence": 0.95,
                  "evidence": ["1x (19.50 Hz = 1170 rpm) dominates at 0.241 px", "2x is only 17% of 1x"]}],
    "severity": None,
    "action": {"kind": "diagnose", "reason": "evidence is clean enough to diagnose", "instructions": []},
}}

REPLY = {"headline": "Mass imbalance on the fan rotor", "diagnosis": "Imbalance: 1x dominates",
         "why": ["1x at 19.5 Hz (1170 rpm) is 0.241 px with 48.8 dB SNR", "2x is only 17% of 1x"],
         "confidence": "high", "next_step": "Balance the rotor",
         "work_order": {"title": "Balance fan", "priority": "P2", "asset": "fan", "tasks": ["Balance"],
                        "parts": [], "estimated_downtime_h": 2}}


@pytest.fixture
def endpoint(monkeypatch):
    monkeypatch.setenv("QUIVER_LLM_BASE_URL", "http://gemma.test/v1")
    monkeypatch.setenv("QUIVER_LLM_MODEL", "google/gemma-3-27b-it")
    calls = []

    def serve(*replies):
        def post(url, json=None, headers=None, timeout=None):
            calls.append(copy.deepcopy(json))  # the retry edits the same body in place
            status, body = replies[min(len(calls), len(replies)) - 1]
            req = httpx.Request("POST", url)
            if isinstance(body, dict):
                body = {"choices": [{"message": {"content": "```json\n" + json_mod.dumps(body) + "\n```"}}]}
            return httpx.Response(status, json=body, request=req)
        monkeypatch.setattr(llm.httpx, "post", post)
        return calls

    return serve


def test_reply_is_parsed_and_every_number_traced(endpoint, tmp_path):
    calls = endpoint((200, REPLY))
    out = llm.write_work_order(RESULT, str(tmp_path), "fan")
    assert out["model"] == "google/gemma-3-27b-it"
    assert out["confidence"] == "high"
    assert out["grounding"]["unverified"] == []
    assert out["grounding"]["checked"] == 5  # 19.5, 1170, 0.241, 48.8, 17
    assert calls[0]["messages"][0]["role"] == "system"


def test_invented_number_is_flagged_and_confidence_dropped(endpoint, tmp_path):
    bad = dict(REPLY, why=["1x at 19.5 Hz is 0.241 px", "bearing defect frequency at 87.3 Hz is rising"])
    endpoint((200, bad))
    out = llm.write_work_order(RESULT, str(tmp_path), "fan")
    assert [u["number"] for u in out["grounding"]["unverified"]] == ["87.3"]
    assert out["confidence"] == "low"


def test_text_only_endpoint_gets_a_retry_without_the_image(endpoint, tmp_path):
    (tmp_path / "heatmap.png").write_bytes(b"\x89PNG fake")
    calls = endpoint((400, {"error": "image input not supported"}), (200, REPLY))
    out = llm.write_work_order(RESULT, str(tmp_path), "fan")
    assert len(calls) == 2
    assert len(calls[0]["messages"][1]["content"]) == 2  # text + image
    assert len(calls[1]["messages"][1]["content"]) == 1  # text only
    assert out["model"] == "google/gemma-3-27b-it"


def test_endpoint_down_falls_back_to_rules(endpoint, tmp_path):
    endpoint((503, {"error": "overloaded"}))
    out = llm.write_work_order(RESULT, str(tmp_path), "fan")
    assert out["model"] == "rules"
    assert "503" in out["llm_error"]
    assert out["work_order"]["title"] == "Mass imbalance on fan"


def test_order_labels_and_rounding_are_not_false_alarms():
    ev = llm._evidence(RESULT, "fan")
    g = llm.grounding({"headline": "1x and 2× dominate", "why": ["running at 19.50 Hz", "about 1170 rpm", "0.24 px"]}, ev)
    assert g["unverified"] == []
    assert g["checked"] == 3


def test_without_an_endpoint_rules_only(monkeypatch, tmp_path):
    monkeypatch.delenv("QUIVER_LLM_BASE_URL", raising=False)
    out = llm.write_work_order(RESULT, str(tmp_path), "fan")
    assert out["model"] == "rules" and "grounding" not in out
