"""Inference engines for the Pupil object-recognition detector server.

Each engine takes a BGR frame and returns a list of *normalized detections*. The server can run
several engines per frame and merge their outputs, so you can e.g. overlay YOLO objects
(cars/people/signs) together with YOLOPv2 road/lane layers in a single pass.

Normalized detection dict:
    {
        "engine": str,                 # "yolo" | "yolopv2" | "sam3"
        "kind":   "object" | "layer",  # object = instance (gaze-eligible); layer = semantic region
        "id":     int | None,          # track id (instances only)
        "name":   str,
        "conf":   float,
        "box":    [x1, y1, x2, y2],    # original-image pixels
        "mask":   [[x, y], ...] | None # polygon, original-image pixels
    }

Engines fail soft: a model that cannot be loaded (missing file, missing package) is skipped with a
warning so the server keeps running with whatever is available. YOLOPv2 and SAM3 are adapted from
the WAMA `cam_analyzer` wrappers.
"""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def poly_bbox(poly):
    arr = np.asarray(poly, dtype=np.float32)
    x1, y1 = arr.min(axis=0)
    x2, y2 = arr.max(axis=0)
    return [float(x1), float(y1), float(x2), float(y2)]


def simplify_polygon(poly, epsilon):
    if epsilon <= 0 or len(poly) < 3:
        return np.asarray(poly).astype(int).tolist()
    contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.reshape(-1, 2).astype(int).tolist()


def mask_to_polygons(mask, min_area, epsilon_ratio=0.005):
    """Binary mask -> list of simplified polygons [[x,y],...] in mask pixel coords."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        eps = max(1.0, epsilon_ratio * cv2.arcLength(c, True))
        c = cv2.approxPolyDP(c, eps, True)
        pts = c.reshape(-1, 2).astype(int).tolist()
        if len(pts) >= 3:
            polys.append(pts)
    return polys


# --------------------------------------------------------------------------------------------------
# temporal smoothing (instances by track id, semantic layers by name)
# --------------------------------------------------------------------------------------------------
class TemporalSmoother:
    """Per-key EMA smoothing of boxes and mask polygons.

    Key = track id for instance objects, or (engine, name) for semantic layers (which may contain
    several disjoint polygons, e.g. multiple lane segments). Masks are rasterised onto a downscaled
    grid, EMA-accumulated, then re-contoured -- robust to the variable vertex count of raw polygons.
    Keys unseen for ``max_age`` frames are forgotten.
    """

    def __init__(self, scale=4, max_age=15):
        self.scale = max(1, scale)
        self.max_age = max_age
        self.dims = None
        self.mask_acc = {}
        self.box_acc = {}
        self.age = {}

    def _ensure_dims(self, shape):
        h = max(1, shape[0] // self.scale)
        w = max(1, shape[1] // self.scale)
        if self.dims != (h, w):
            self.dims = (h, w)
            self.mask_acc.clear()
            self.box_acc.clear()
            self.age.clear()

    @staticmethod
    def _key(det):
        if det.get("id") is not None:
            return ("id", det["id"])
        return ("sem", det.get("engine", ""), det["name"])

    def smooth(self, detections, shape, alpha):
        if alpha <= 0:
            return detections
        alpha = min(alpha, 0.95)
        self._ensure_dims(shape)
        h, w = self.dims
        s = self.scale

        groups = {}
        for det in detections:
            groups.setdefault(self._key(det), []).append(det)

        out, seen = [], set()
        for key, group in groups.items():
            seen.add(key)
            template = group[0]
            has_mask = any(d.get("mask") for d in group)

            if has_mask:
                canvas = np.zeros((h, w), dtype=np.float32)
                for d in group:
                    if d.get("mask"):
                        poly = (np.asarray(d["mask"], dtype=np.float32) / s).astype(np.int32).reshape(-1, 1, 2)
                        cv2.fillPoly(canvas, [poly], 1.0)
                acc = self.mask_acc.get(key)
                acc = canvas if acc is None else alpha * acc + (1.0 - alpha) * canvas
                self.mask_acc[key] = acc

                binary = (acc >= 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) < 1:
                        continue
                    c = (c.astype(np.float32) * s)
                    c = cv2.approxPolyDP(c, max(1.0, s * 0.75), True)
                    poly = c.reshape(-1, 2).astype(int).tolist()
                    if len(poly) < 3:
                        continue
                    nd = dict(template)
                    nd["mask"] = poly
                    nd["box"] = poly_bbox(poly)
                    out.append(nd)
            else:
                # box-only smoothing (e.g. detection-only YOLO)
                box = np.asarray(template["box"], dtype=np.float32)
                acc = self.box_acc.get(key)
                acc = box if acc is None else alpha * acc + (1.0 - alpha) * box
                self.box_acc[key] = acc
                nd = dict(template)
                nd["box"] = acc.tolist()
                out.append(nd)

        for key in list(set(self.mask_acc) | set(self.box_acc)):
            if key in seen:
                self.age[key] = 0
            else:
                self.age[key] = self.age.get(key, 0) + 1
                if self.age[key] > self.max_age:
                    self.mask_acc.pop(key, None)
                    self.box_acc.pop(key, None)
                    self.age.pop(key, None)
        return out


# --------------------------------------------------------------------------------------------------
# engines
# --------------------------------------------------------------------------------------------------
class UltralyticsEngine:
    """YOLO detection or segmentation with ByteTrack tracking (real-time)."""

    name = "yolo"
    realtime = True

    def __init__(self, model_path="yolo11n-seg.pt", imgsz=640, device=None, half=False,
                 track=True, tracker="bytetrack.yaml", mask_epsilon=1.5):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.task = getattr(self.model, "task", "detect")
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.track = track
        self.tracker = tracker
        self.mask_epsilon = mask_epsilon
        logger.info("[yolo] loaded %s (task=%s, track=%s)", model_path, self.task, track)

    def infer(self, img, conf):
        if self.track:
            results = self.model.track(img, persist=True, tracker=self.tracker, conf=conf,
                                       imgsz=self.imgsz, device=self.device, half=self.half,
                                       verbose=False)
        else:
            results = self.model.predict(img, conf=conf, imgsz=self.imgsz, device=self.device,
                                         half=self.half, verbose=False)
        r = results[0]
        boxes = r.boxes
        if boxes is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        cf = boxes.conf.cpu().numpy()
        ids = boxes.id
        ids = ids.cpu().numpy().astype(int) if ids is not None else None
        polys = r.masks.xy if r.masks is not None else None
        dets = []
        for i in range(len(xyxy)):
            mask = None
            if polys is not None and i < len(polys) and len(polys[i]) >= 3:
                mask = simplify_polygon(polys[i], self.mask_epsilon)
            dets.append({
                "engine": "yolo", "kind": "object",
                "id": int(ids[i]) if ids is not None else None,
                "name": str(r.names[int(cls[i])]), "conf": float(cf[i]),
                "box": [float(v) for v in xyxy[i]], "mask": mask,
            })
        return dets


# YOLOPv2 official input size (height, width) and constants (from CAIC-AD/YOLOPv2 demo).
_YP_H, _YP_W = 384, 640


def _letterbox(img, new_h, new_w):
    h, w = img.shape[:2]
    r = min(new_h / h, new_w / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    pad_w = (new_w - nw) // 2
    pad_h = (new_h - nh) // 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
    out[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    return out, r, pad_w, pad_h


class YOLOPv2Engine:
    """YOLOPv2 TorchScript model: drivable-area + lane-line semantic segmentation (real-time).

    Adapted from the WAMA cam_analyzer YOLOPv2RoadSegmenter. Object-detection head is ignored
    (YOLO handles instances). Produces semantic 'layer' detections.
    """

    name = "yolopv2"
    realtime = True

    def __init__(self, model_path="yolopv2.pt", device=None):
        import torch
        self.torch = torch
        dev = device
        if dev in (None, "", "auto"):
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev != "cpu" and not torch.cuda.is_available():
            logger.warning("[yolopv2] CUDA unavailable -> CPU")
            dev = "cpu"
        self.device = dev
        self.model = torch.jit.load(model_path, map_location=dev)
        self.model.eval()
        with torch.no_grad():
            try:
                self.model(torch.zeros(1, 3, _YP_H, _YP_W, device=dev))
            except Exception as exc:
                logger.warning("[yolopv2] warmup failed (non-fatal): %s", exc)
        logger.info("[yolopv2] loaded %s on %s", model_path, dev)

    def _decode_mask(self, t, pad_w, pad_h, w0, h0):
        if t is None:
            return None
        if t.dim() == 4 and t.shape[1] == 2:
            m = self.torch.argmax(t, dim=1).squeeze(0)
        else:
            m = (t.squeeze() > 0.5).int()
        mask = m.detach().cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (_YP_W, _YP_H), interpolation=cv2.INTER_NEAREST)
        mask = mask[pad_h:_YP_H - pad_h if pad_h else _YP_H,
                    pad_w:_YP_W - pad_w if pad_w else _YP_W]
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
        return mask

    def infer(self, img, conf):
        h0, w0 = img.shape[:2]
        lb, _, pad_w, pad_h = _letterbox(img, _YP_H, _YP_W)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).to(self.device).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        try:
            with self.torch.no_grad():
                out = self.model(tensor)
        except Exception as exc:
            logger.warning("[yolopv2] inference failed: %s", exc)
            return []

        seg, ll = None, None
        if isinstance(out, (list, tuple)):
            if len(out) >= 3:
                seg, ll = out[1], out[2]
            elif len(out) == 2:
                seg, ll = out
        if seg is None and ll is None:
            return []

        dets = []
        for mask_t, label, min_area in ((seg, "drivable area", 2000), (ll, "lane", 200)):
            mask = self._decode_mask(mask_t, pad_w, pad_h, w0, h0)
            if mask is None:
                continue
            for poly in mask_to_polygons(mask, min_area=min_area):
                dets.append({
                    "engine": "yolopv2", "kind": "layer", "id": None,
                    "name": label, "conf": 1.0, "box": poly_bbox(poly), "mask": poly,
                })
        return dets


class SAM3Engine:
    """SAM3 text-prompt segmentation (offline-grade; slow frame-by-frame on Windows).

    Adapted from the WAMA SAM3RoadAnalyzer. Lazy-imports the `sam3` package so the server runs fine
    when SAM3 is not installed. Each prompt produces masks; markings become 'object' detections,
    the road prompt becomes a 'layer'. Intended mainly for the Pupil Player (offline) path.
    """

    name = "sam3"
    realtime = False

    def __init__(self, prompts=None, road_prompt=None, conf_threshold=0.30, hf_setup=None):
        self.prompts = prompts or [
            {"label": "stop_line", "prompt": "white stop line painted on road surface"},
            {"label": "crossing", "prompt": "pedestrian crossing zebra stripes on road"},
        ]
        self.road_prompt = road_prompt  # e.g. "drivable road surface area in front of the vehicle"
        self.conf_threshold = conf_threshold
        if hf_setup is not None:
            hf_setup()  # configure HF env/token before importing sam3
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self._processor = Sam3Processor(build_sam3_image_model())
        logger.info("[sam3] image model loaded (%d marking prompts, road=%s)",
                    len(self.prompts), bool(self.road_prompt))

    @staticmethod
    def _mask_to_numpy(mask, hw):
        try:
            import torch
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
        except ImportError:
            pass
        mask = np.asarray(mask)
        while mask.ndim > 2:
            mask = mask.squeeze(0) if mask.shape[0] == 1 else mask.squeeze()
        mask = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
        if mask.shape[:2] != hw:
            mask = cv2.resize(mask, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
        return mask

    def infer(self, img, conf):
        from PIL import Image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        h, w = img.shape[:2]
        dets = []

        def _run(prompt):
            state = self._processor.set_image(pil)
            output = self._processor.set_text_prompt(state=state, prompt=prompt)
            return output.get("masks", []) or [], output.get("scores", []) or []

        for pd in self.prompts:
            prompt = (pd.get("prompt") or "").strip()
            if not prompt:
                continue
            try:
                masks, scores = _run(prompt)
            except Exception as exc:
                logger.debug("[sam3] prompt '%s' failed: %s", pd.get("label"), exc)
                continue
            for i, mask in enumerate(masks):
                score = float(scores[i]) if i < len(scores) else 1.0
                if score < self.conf_threshold:
                    continue
                m = self._mask_to_numpy(mask, (h, w))
                polys = mask_to_polygons((m > 127).astype(np.uint8), min_area=100)
                poly = max(polys, key=cv2.contourArea) if polys else None
                box = poly_bbox(poly) if poly else [0, 0, 0, 0]
                dets.append({
                    "engine": "sam3", "kind": "object", "id": None,
                    "name": pd.get("label", "marking"), "conf": round(score, 3),
                    "box": box, "mask": poly,
                })

        if self.road_prompt:
            try:
                masks, scores = _run(self.road_prompt)
            except Exception as exc:
                logger.debug("[sam3] road prompt failed: %s", exc)
                masks, scores = [], []
            for i, mask in enumerate(masks):
                score = float(scores[i]) if i < len(scores) else 1.0
                if score < self.conf_threshold:
                    continue
                m = self._mask_to_numpy(mask, (h, w))
                polys = mask_to_polygons((m > 127).astype(np.uint8), min_area=500)
                if not polys:
                    continue
                poly = max(polys, key=cv2.contourArea)
                dets.append({
                    "engine": "sam3", "kind": "layer", "id": None,
                    "name": "drivable area (sam3)", "conf": round(score, 3),
                    "box": poly_bbox(poly), "mask": poly,
                })
                break

        return dets


def build_engines(args):
    """Instantiate the engines named in ``args.engines`` (comma-separated). Fail-soft."""
    requested = [e.strip().lower() for e in args.engines.split(",") if e.strip()]
    engines = []
    for name in requested:
        try:
            if name == "yolo":
                engines.append(UltralyticsEngine(
                    model_path=args.model, imgsz=args.imgsz, device=args.device, half=args.half,
                    track=not args.no_track, tracker=args.tracker, mask_epsilon=args.mask_epsilon))
            elif name == "yolopv2":
                engines.append(YOLOPv2Engine(model_path=args.yolopv2_model, device=args.device))
            elif name == "sam3":
                prompts = None
                if args.sam3_prompts:
                    prompts = [{"label": p.strip().replace(" ", "_"), "prompt": p.strip()}
                               for p in args.sam3_prompts.split(";") if p.strip()]
                engines.append(SAM3Engine(prompts=prompts, road_prompt=args.sam3_road or None))
            else:
                logger.warning("unknown engine '%s' (skipped)", name)
        except Exception as exc:
            logger.error("could not load engine '%s' -> skipped: %s", name, exc)
    if not engines:
        raise RuntimeError("no engine could be loaded (requested: %s)" % args.engines)
    return engines
