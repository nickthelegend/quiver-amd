# Quiver — lablab submission kit (AMD Developer Hackathon: ACT III)

**Deadline: Oct 18, 2026, 3:30 PM IST** (lablab page). Track: **1 · Intelligent Industry**.
Partner prize: **Google** (Gemma on AMD).

Fill in every `{{…}}` from the real fan recordings before submitting.

---

## Project title
Quiver: hear a machine from a silent video

## Short description (one line)
Film a running machine in slow motion and Quiver recovers its vibration, and even its sound, from
the pixels. Gemma on AMD Instinct then diagnoses the fault and writes the work order.

## Long description

Vibration analysis catches failing motors, fans and pumps weeks before they break, but it needs
sensors bolted to every machine or specialist camera systems sold by quote. Most mid-size plants
check a handful of critical assets and run the rest to failure.

Quiver turns the phone a technician already carries into a first-pass vibration analyser:

1. **Film** the machine for a few seconds in slow motion (240 fps), phone on a tripod.
2. **Measure.** OpenCV tracks every region of the frame to a few hundredths of a pixel
   (Lucas–Kanade against the clip's mean frame), removes camera shake, and builds the vibration
   spectrum.
3. **See and hear it.**
   - Quiver plays back the sound the machine is making, recovered from a *silent* video.
   - It shows the shaking, magnified ×30 or more, with the background held still.
4. **Diagnose with evidence.** The spectrum is read the way a vibration analyst reads it:

   | Pattern | Diagnosis |
   |---|---|
   | 1× running speed dominant | imbalance |
   | strong 2× | misalignment |
   | a run of harmonics | looseness |
   | sub-synchronous peak | rub / whirl |

   Every finding cites its numbers.
5. **Act.** **Gemma, served on AMD Instinct GPUs**, receives the measured evidence plus a heatmap of
   where the machine vibrates. It writes the diagnosis and a maintenance work order, citing
   measured numbers only, and Quiver checks that it did. Every number in Gemma's reasoning is
   traced back to a measurement (a frequency, its rpm, an amplitude, a ratio between orders). If
   one can't be traced, it is flagged on screen and the confidence drops to low. On a clean
   reply the page says "all 5 numbers traced to measurements".

**It's an agent, not a report.** When the evidence isn't good enough, Quiver doesn't guess.
- If the signal is weak, the camera shook, the clip is too short, or the frame rate is too low to
  see 2× running speed, it says exactly what to re-film.
- When a second recording arrives at another frame rate, it works out which true frequency is
  consistent with both (alias resolution).
- If vibration is in the damage zone, it stops and asks a supervisor before the machine keeps
  running.

**Results.**
- **Synthetic ground truth** (clips rendered with a known vibration):
  - running speed 19.50 Hz vs a true 19.5 Hz
  - amplitude within 7%
  - misalignment, looseness, a still machine and 30 fps aliasing all handled
- **Real fan:** {{1× peak Hz}} vs phone accelerometer {{phyphox Hz}}; imbalance detected with a coin
  on one blade ({{1× amplitude with coin}} vs {{without}}).

**Where AMD fits:** Gemma {{model}} runs on AMD Instinct {{MI300X}} via {{Token Factory / vLLM on
ROCm}} and writes every diagnosis and work order. The vision pipeline is CPU-light OpenCV 5, so
one GPU can serve a whole plant's technicians.

**What's next:** CMMS integration (Maximo, Fiix, UpKeep), route-based inspections, trending each
asset over time.

## Technology tags
AMD Instinct, ROCm, AMD Developer Cloud, Gemma, Google DeepMind, OpenCV, Computer Vision, Predictive
Maintenance, Signal Processing, FastAPI, Python, Agents

## Category / challenge
Track 1: Intelligent Industry (plus Google partner prize)

## Links
- Repo: https://github.com/nickthelegend/quiver-amd (**make public before submitting**)
- App URL: {{deployed URL}}
- Video: {{YouTube/Vimeo URL}}

---

## Demo video script (≤ 3 min)

| Time | Picture | Voice-over |
|---|---|---|
| 0:00–0:12 | Desk fan running. Then the phone screen recording slow-mo, sound muted (show the mute icon). | "This video has no sound. Watch." |
| 0:12–0:25 | Quiver result page: press **True pitch**, then **Lifted ×N**. The hum plays. | "That's the fan, recovered from the pixels: every part of the frame tracked to a hundredth of a pixel." |
| 0:25–0:45 | Magnified video side by side: original still, magnified wobbling. | "Here's what it's doing: magnified thirty times. Something's wrong." |
| 0:45–1:15 | Spectrum: hover the 1× peak; the 2× marker small. Diagnosis card: *mass imbalance*. Work order card. | "One big peak at running speed, {{rpm}} rpm, and little else. That's imbalance. Gemma, running on AMD Instinct, writes the work order and has to cite every number it uses." |
| 1:15–1:35 | Reveal the coin taped to the blade. Remove it, re-film, upload: "no significant fault". | "The cause: a coin on one blade. Take it off: healthy." |
| 1:35–1:55 | Overlay chart: Quiver's spectrum vs the phyphox accelerometer taped to the fan. | "Checked against a real accelerometer: same peak." |
| 1:55–2:25 | Upload a 30 fps clip → amber "needs a better recording" panel with the instructions → upload the 240 fps re-recording → alias check chip. | "When the evidence is weak it won't guess. It tells you what to re-film, then checks the second clip against the first." |
| 2:25–2:50 | Architecture diagram: phone → OpenCV motion → spectrum/rules → Gemma on AMD → work order. | "OpenCV does the measuring; Gemma on AMD Instinct does the reasoning. Any phone, any plant." |
| 2:50–3:00 | Title card. | "Quiver. Hear what your machines are telling you." |

## Slide outline (8 slides)
1. **Quiver**: hear a machine from a silent video (hero: the magnified fan frame)
2. **The problem**: vibration catches failures early, but sensors on every asset are too expensive, so most machines run to failure
3. **The insight**: a slow-motion camera is a grid of thousands of vibration sensors
4. **How it works**: film → sub-pixel motion → spectrum → rules → Gemma on AMD → work order (diagram)
5. **Demo results**: real fan, coin on/off, accelerometer overlay; synthetic ground truth table
6. **An agent, not a report**: re-film requests, alias resolution, supervisor approval for damage-zone readings
7. **AMD**: Gemma on Instinct via ROCm; why GPU reasoning and CPU vision split cleanly
8. **Business**: route-based inspection SaaS for mid-size plants; CMMS integration; next steps

## Cover image
The side-by-side magnified frame (original | magnified) with the spectrum peak overlaid and the
word QUIVER. Export from a real run: `runs/<id>/magnified.mp4` frame + `heatmap.png`.

## Before you hit submit
- [ ] Repo public, README runs, no keys committed (`.env` is git-ignored)
- [ ] App deployed and reachable; Gemma endpoint live on AMD
- [ ] Video ≤ 3 min, public/unlisted, shows AMD usage explicitly
- [ ] Slides uploaded (PDF)
- [ ] Track 1 selected; Google/Gemma prize opted in
