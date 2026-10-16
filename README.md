# Quiver

**Hear a machine from a silent video.** Film a running motor, fan or pump in slow
motion with a phone. Quiver measures how every part of the image moves, to a few
hundredths of a pixel. From that it gives you:

- **the sound** the machine is making, recovered from pixels (the clip can be silent)
- **a magnified video** in which invisible shaking becomes a visible wobble
- **a vibration spectrum** with running speed and its multiples (1×, 2×, 3×…) marked
- **a diagnosis with evidence**: imbalance, misalignment, looseness and more, each
  backed by a measured number
- **a maintenance work order**, written by Gemma running on AMD Instinct GPUs

When the footage isn't good enough, Quiver doesn't guess. It says what to re-film
(closer, steadier, higher frame rate) and checks the second recording against the
first.

Built for the **AMD Developer Hackathon: ACT III**, Track 1 · Intelligent Industry.

## Why this matters

Vibration analysis catches failing machines weeks early, but it needs sensors bolted
to every asset or specialist high-speed camera systems. Quiver turns the phone a
technician already carries into a first-pass vibration analyser.

## How it works

```
phone video ──► tiles ──► Lucas-Kanade sub-pixel shift per tile, per frame
                              │
                    camera shake removed (median of static tiles)
                              │
          vibrating tiles combined ──► FFT spectrum ──► order analysis (1x, 2x, 3x…)
                              │                              │
           sound (true pitch + lifted)                fault rules + quality checks
           magnified video (Lagrangian)                       │
                                                agent: diagnose / re-record / escalate
                                                              │
                                         Gemma on AMD (vision + text) ──► work order
```

- **Motion.** Each frame is split into a 6×8 grid. Against the clip's mean frame, a
  gradient (Lucas–Kanade) solve gives each tile's shift per frame. Untrackable,
  textureless tiles are ignored.
- **Spectrum.** Motion is projected on its main direction, then windowed FFT with
  sub-bin peak refinement. Amplitudes are in pixels of image motion. If you give a
  mm-per-pixel scale, Quiver converts to mm/s RMS and an ISO 10816 zone.
- **Rules.**

  | Pattern | Diagnosis |
  |---|---|
  | 1× dominant | imbalance |
  | 2× ≥ 50% of 1× | misalignment |
  | 2+ harmonics including 3× or above | looseness |
  | peak near 0.5× | rub / whirl |
  | peak not a multiple of running speed | another source |

- **Quality checks.** These decide what happens next: signal too weak, camera shake,
  clip too short, or a frame rate too low to see 2× (or low enough that the peak may
  be aliased). Recording the same view at a second frame rate settles the aliasing.
- **AI technician.** Gemma receives the measured evidence and the heatmap image. It
  must cite numbers, and it writes a JSON work order. Without an endpoint, a
  rules-only work order is used.

## Ground truth

`tools/make_synthetic.py` renders clips with a known vibration. `pytest` checks
Quiver recovers it:

| Case | Truth | Quiver |
|---|---|---|
| Running speed | 19.5 Hz | 19.50 Hz |
| 1× amplitude | 0.225 px | 0.241 px |
| 2× amplitude | 0.0375 px | 0.042 px |
| Misalignment / looseness / still machine / 30 fps aliasing | — | detected / detected / asks to re-film / asks for slow motion |

On real footage, tape a phone running the free **phyphox** accelerometer app to the
machine and compare its spectrum with Quiver's.

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # point it at Gemma on AMD (see below), or skip for rules-only
.venv/bin/uvicorn server:app --port 8710
# open http://localhost:8710
```

Tests: `.venv/bin/python -m pytest -q`

Docker: `docker build -t quiver . && docker run -p 8000:8000 --env-file .env quiver`

## Gemma on AMD

Quiver talks to any OpenAI-compatible endpoint. On the AMD Developer Cloud, either:

- use the event's **Token Factory** endpoint and key, or
- serve Gemma yourself with vLLM on an Instinct GPU (ROCm):

```bash
vllm serve google/gemma-3-27b-it --max-model-len 8192 --port 8000
```

Then set `QUIVER_LLM_BASE_URL`, `QUIVER_LLM_API_KEY` and `QUIVER_LLM_MODEL`.

## Filming tips

- **Hold it still.** Use a tripod, or brace the phone on something solid.
- **Shoot slow motion** (240 fps) for 4–10 seconds.
- **Frame it tight.** The machine housing should fill a good part of the frame.
- **Add texture.** On a plain, shiny housing, add a strip of high-contrast tape.

## Limits

- At 240 fps Quiver sees up to 120 Hz. That covers running speed and its first
  harmonics for most fans, pumps and motors, but not speech or high-frequency
  bearing tones.
- Amplitudes are image motion unless you calibrate the scale.
- The rules are a first-pass screen, not a replacement for a vibration analyst.

## License

MIT
