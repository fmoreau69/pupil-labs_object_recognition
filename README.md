# Pupil Labs — Object Recognition

Real-time **object detection & segmentation** plugin for [Pupil Core](https://docs.pupil-labs.com/core/)
(Pupil Capture **and** Pupil Player), built for cognitive-science driving experiments: from the
world camera of a participant wearing a Pupil Core headset, it identifies the objects in the scene
and determines **what the participant is looking at** — not just *where* they look.

> **2026 rewrite.** Darknet and imagezmq are gone. Detection now runs on
> [Ultralytics YOLO](https://docs.ultralytics.com/) with optional tracking and segmentation, in a
> separate process, so the plugin installs as a **single `.py` file** and stays real-time.

---

## What it does

- **Detects / segments** objects in the world view (YOLO, COCO classes by default).
- **Tracks** objects across frames (stable ids) and **temporally smooths** boxes/masks so contours
  don't flicker.
- **Gaze ↔ object matching**: the object under the gaze is drawn in **red** (observed), the others
  in **green** — using point-in-box *or* point-in-mask.
- **Records** the object data natively into the Pupil recording (`objects.pldata` + `objects.csv`)
  and publishes it on the Pupil IPC backbone (topic `objects`).
- **Streams** the object data to **RTMaps** and **LSL** for multi-sensor synchronisation.
- **Pupil Player** plugin to replay the overlay and to **reprocess raw recordings** offline
  (re-run detection + gaze matching on sessions acquired without the plugin).
- Optional extra engines: **YOLOPv2** (drivable area / lane lines) and **SAM3** (text-prompt
  segmentation, offline).

---

## Architecture

The Pupil Capture/Player bundle ships **Python 3.6**, which can't run PyTorch. So detection runs in
a separate **Python 3.12** process; the in-bundle plugin is a thin client.

```
┌─ Pupil Capture / Player (bundle, Python 3.6) ─┐        ┌─ Detector (your venv, Python 3.12) ─┐
│  capture_/player_object_recognition.py        │  ZMQ   │  yolo_server.py + engines.py        │
│  • world frame + gaze                         │ ─────► │  • ultralytics YOLO (+track/smooth) │
│  • overlay, gaze matching (red/green)         │ ◄───── │  • yolopv2 / sam3 (optional)        │
│  • record objects.pldata + objects.csv        │        └─────────────────────────────────────┘
│  • IPC "objects" + ZMQ PUB export             │
└──────┬─────────────────────────────────────────┘
       │ ZMQ PUB (object data)
       ├──► RTMaps   (rtmaps_stream.py)
       └──► LSL      (lsl_relay.py → LabRecorder)
```

---

## Requirements

- **Pupil Core** app bundle (Pupil Capture / Player) — installed from the official `.msi`/installer.
- **Python 3.12** with an **NVIDIA GPU + CUDA** for real-time detection (CPU works but slower).
- The detector deps install via pip (next section). The plugins need **no extra install** — they
  use only what the bundle already ships (pyzmq, numpy, opencv, msgpack, pyglui).

---

## Installation

### Quick install (Windows, recommended)

With the Pupil Core bundle already installed and this repo cloned, just run the installer
(double-click **`install.bat`**, or from PowerShell):

```powershell
.\install.ps1            # auto: detects the NVIDIA GPU and picks the right torch build
.\install.ps1 -Cpu       # force the CPU build (laptop without an NVIDIA GPU)
.\install.ps1 -NoServer  # install only, don't launch the detector at the end
```

It creates the `.venv`, **detects the GPU** (and warns about a CPU fallback if there's no NVIDIA
card), installs the dependencies, copies both plugins into `~/pupil_capture_settings/plugins` and
`~/pupil_player_settings/plugins` (creating the folders if Capture/Player were never launched), then
starts the detector. Afterwards, start the detector any time with **`start_detector.bat`**.

The manual steps below are equivalent, for reference or other platforms.

### 1. Detector (once, Python 3.12 venv)

Install the **CUDA** build of torch *first* (otherwise pip pulls the CPU build on Windows):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r detector/requirements-detector.txt
```

Check the GPU is visible:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Plugins (just copy one file each)

- **Capture** → copy `plugins/capture_object_recognition.py` into `~/pupil_capture_settings/plugins/`
- **Player**  → copy `plugins/player_object_recognition.py` into `~/pupil_player_settings/plugins/`

(`~` is your home folder, e.g. `C:\Users\<you>\`.)

### 3. Optional — multi-sensor export

- **RTMaps**: put `integrations/rtmaps_stream.py` (data) and/or `integrations/rtmaps_video.py`
  (annotated video) in an RTMaps Python block.
- **LSL**: in the detector venv, `pip install -r integrations/requirements-relay.txt` (adds `pylsl`).

---

## Repository layout

```
install.bat / install.ps1   Windows one-shot installer (venv + deps + plugins + server)
start_detector.bat          launch the detector later (reuses the .venv)
plugins/        capture_object_recognition.py, player_object_recognition.py   ← single-file drop-ins
detector/       yolo_server.py, engines.py, requirements-detector.txt
integrations/   rtmaps_stream.py, rtmaps_video.py, lsl_relay.py, requirements-relay.txt
                RTMaps/   example RTMaps acquisition diagrams (.rtd)
models/         downloaded model weights (git-ignored)
```

---

## Usage

### Start the detector

```powershell
python detector/yolo_server.py                       # segmentation (yolo11n-seg), default
python detector/yolo_server.py --model yolo11n.pt    # detection only (lighter, e.g. weaker laptop)
```
Leave it running. The weights (~6 MB) download into `models/` automatically on first launch.

### Pupil Capture

1. Start Pupil Capture, open **Plugin Manager**, enable **Object Recognition (YOLO)**.
2. Toggle **Launch object recognition**. Boxes/masks appear; after calibration the looked-at object
   turns red.

**Controls**

| Control | What it does |
|---|---|
| Detector address | Where the detector listens (default `tcp://127.0.0.1:5560`). |
| Display confidence | Hides objects below this score **from the overlay & "observed" pick**. Detection still finds everything; all detections are recorded. |
| Temporal smoothing | 0–0.95. Higher = steadier contours, slightly more lag. |
| Max detection rate | Caps inference rate (Hz); the overlay holds between frames. Lower it to steady the view or save GPU. |
| Classes to display | Comma-separated whitelist (e.g. `person, car, bus, traffic light`); empty = all. Display-only; everything is still recorded. |
| Draw segmentation masks | Outline masks instead of boxes (with a seg model). |
| Show track ids | Prefix labels with the track id (`#3`). `#-` means the detector returned no id → it's not tracking (restart `yolo_server.py` without `--no-track`). |
| Only observed object | Draw/keep only the looked-at object. |
| Stream object data (RTMaps/LSL) | Publish the per-frame object data on a ZMQ PUB socket (see Exports). |
| Stream annotated video (RTMaps) | Publish the overlaid world frame (JPEG) on a separate PUB socket for RTMaps. |
| Record annotated video | Write `world_overlay.mp4` into the recording folder during a Pupil recording. |

### Pupil Player

Open a recording with **Object Recognition (YOLO) — Player** enabled:

- **Replay**: if the recording already has `objects.pldata` (recorded live, or from a previous
  reprocess), the overlay is redrawn automatically. Same display controls as Capture.
- **Reprocess**: for raw recordings (world video + gaze, no object data), click **Reprocess
  recording** (detector running). It runs detection + gaze matching over every frame and writes
  `objects.pldata` + `objects.csv`. Use this to analyse eye-tracking data acquired *without* the
  plugin, or to run heavy/offline engines (SAM3). ⚠️ overwrites any existing `objects.pldata`.

### Exports (RTMaps / LSL)

Enable **Stream object data (RTMaps/LSL)** in the Capture plugin (bind address `tcp://*:5561`
exposes it on the LAN; `tcp://127.0.0.1:5561` is localhost only). Then:

- **RTMaps**: set the `integrations/rtmaps_stream.py` block's `sub_address` to
  `tcp://<pupil-host>:5561`. Outputs: observed name/id/box, gaze, object count, timestamp, JSON.
- **LSL**: `python integrations/lsl_relay.py --connect tcp://<pupil-host>:5561`. Creates two LSL outlets —
  `PupilObjects` (numeric: `observed, x1, y1, x2, y2, gaze_x, gaze_y`) and `PupilObjects_json`
  (full datum as a string marker) — recordable in LabRecorder.

**Annotated video** (overlay burned in) — enable **Stream annotated video (RTMaps)** and/or
**Record annotated video**:

- **RTMaps**: set the `integrations/rtmaps_video.py` block's `sub_address` to
  `tcp://<pupil-host>:5562` to view the overlaid world camera live in RTMaps.
- **File**: with a Pupil recording running, `world_overlay.mp4` is written into the recording
  folder. (For Player you usually don't need it — the overlay is rebuilt from `objects.pldata`.)

Example RTMaps diagrams live in `integrations/RTMaps/`. Their PythonBridge `pythonFilename` still
points at old absolute paths — repoint it to `integrations/rtmaps_video.py` (annotated video) or
`integrations/rtmaps_stream.py` (object data) when you open them.

> **Offline note**: LSL/RTMaps are *live* sync transports. If you detect only in post-processing,
> use `objects.pldata` / `objects.csv` and merge with your live LSL/XDF + RTMaps logs **by
> timestamp** — don't stream offline (the replay clock ≠ session clock).

---

## Data outputs

| Output | Where | Contents |
|---|---|---|
| `objects.pldata` | recording folder | Full per-frame data (Pupil-native, re-loadable in Player). |
| `objects.csv` | recording folder | Flat per-frame row: frame, timestamp, gaze, observed name/id/box, object count. |
| `world_overlay.mp4` | recording folder | Annotated world video (optional, "Record annotated video"). |
| IPC topic `objects` | Pupil IPC backbone | Live, for other Pupil plugins / network clients. |
| ZMQ PUB (`tcp://*:5561`) | network | `[b"objects", msgpack(datum)]` per frame → RTMaps / LSL relay. |
| ZMQ PUB (`tcp://*:5562`) | network | `[b"frame", jpg]` annotated video → RTMaps (`rtmaps_video.py`). |

Per-frame **datum**: `{topic, timestamp, frame_index, gaze_2d, focus:{id,name,conf,box,mask},
objects:[{id,kind,engine,name,conf,box}]}`. `kind` is `object` (instance) or `layer` (semantic
region like road/lane).

---

## Engines (advanced)

Run several at once with `--engines` (merged into one overlay):

```powershell
python detector/yolo_server.py --engines yolo,yolopv2 --yolopv2-model models/yolopv2.pt
python detector/yolo_server.py --engines sam3 --sam3-road "drivable road surface in front of the vehicle"
```

| Engine | Output | Real-time | Needs |
|---|---|:--:|---|
| `yolo` | objects + masks, tracking | ✅ | ultralytics (installed) |
| `yolopv2` | drivable-area + lane layers | ✅ | `models/yolopv2.pt` (TorchScript) |
| `sam3` | text-prompt masks | ❌ (offline) | `sam3` pkg + gated HF `facebook/sam3` |

SAM3 is offline-grade on Windows (its video predictor needs Linux/`triton`) — use it in Player.

---

## Troubleshooting

- **"Detector: DISCONNECTED"** → start `python detector/yolo_server.py`; check the **Detector address**.
- **Track ids show `#-`** → the detector isn't tracking; restart it (tracking is on by default,
  don't pass `--no-track`).
- **Overlay too jittery** → raise **Temporal smoothing** (0.6–0.8) and/or lower **Max detection rate**.
- **Raising confidence doesn't reduce boxes** → use **Display confidence** (it filters the overlay
  deterministically; the server keeps detecting/recording everything).

---

## Roadmap & credits

See [`ROADMAP.md`](ROADMAP.md) for architecture details and the remaining work (annotated-video
streaming, live validation of YOLOPv2/SAM3).

Originally developed in 2020 by **Baptiste Broyer** under the supervision of **Fabien Moreau** at
**Université Gustave Eiffel** (LESCOT, Lyon-Bron). 2026 rewrite to Ultralytics + multi-process
architecture.

Contact: fabien.moreau@univ-eiffel.fr
