"""
Object Recognition (YOLO) — Pupil **Player** plugin.

Offline companion to the Capture plugin. Two modes:

  1. **Replay**  — if the recording already contains ``objects.pldata`` (written live by the Capture
     plugin, or by a previous reprocess), it loads it and redraws the overlay on the recorded world
     video, frame by frame. No inference.

  2. **Reprocess** — for *raw* recordings (world video + gaze, but no ``objects.pldata``), it
     iterates over every world frame, runs detection through the same external detector
     (``yolo_server.py``) over ZMQ, re-does the gaze↔object matching, and writes ``objects.pldata``
     into the recording. This lets you analyse eye-tracking data that was acquired without the
     plugin, and is the right place for heavy/offline engines like SAM3.

Runs in the Pupil Player bundle (Python 3.6); only deps already present are used: pyzmq, numpy,
opencv, msgpack, pyglui. The external detector must be running (``python yolo_server.py``).
"""
import glob
import logging
import os
import threading
import time

import cv2
import msgpack
import numpy as np
import zmq
from pyglui import ui
from pyglui.cygl.utils import draw_gl_texture

import gl_utils
from plugin import Plugin

try:
    from file_methods import PLData_Writer, load_pldata_file
except Exception:  # pragma: no cover - API guard
    PLData_Writer = None
    load_pldata_file = None

logger = logging.getLogger(__name__)

# Same palette as the Capture plugin (BGR).
COLOR_OBSERVED = (12, 15, 145)
COLOR_OTHER = (172, 183, 114)
LAYER_COLORS = {
    "drivable area": (180, 200, 90),
    "drivable area (sam3)": (180, 200, 90),
    "lane": (40, 200, 255),
}
LAYER_DEFAULT_COLOR = (200, 120, 200)
LAYER_ALPHA = 0.35


class Object_Recognition_Player(Plugin):
    """Replays recorded object data over the world video, and can reprocess raw recordings."""

    icon_chr = "O"

    def __init__(self, g_pool, server_address="tcp://127.0.0.1:5560", conf=0.25, smooth=0.5,
                 sight_only=False, show_masks=True, show_ids=False, classes_filter=""):
        super().__init__(g_pool)
        self.order = 0.1

        self.server_address = server_address
        self.conf = conf
        self.smooth = smooth
        self.sight_only = sight_only
        self.show_masks = show_masks
        self.show_ids = show_ids
        self.classes_filter = classes_filter

        # runtime
        self.menu = None
        self.status_text = None
        self.classes_text = None
        self.img = None
        self.width, self.height = 1280, 720
        self.gaze_position_2d = []
        self._whitelist = frozenset()
        self._seen_classes = set()
        self._index = {}              # frame_index -> datum (replay)
        self._reproc = None           # background reprocess state dict

        self.load_objects_data()

    # --- UI -------------------------------------------------------------------------------
    def init_ui(self):
        self.add_menu()
        self.menu.label = "Object Recognition (YOLO) — Player"
        self.menu.append(ui.Info_Text(
            "Replays recorded object data (objects.pldata) over the world video. If the recording "
            "has none, use 'Reprocess' to run detection + gaze matching offline (needs the external "
            "detector running: `python yolo_server.py`)."
        ))
        self.menu.append(ui.Text_Input("classes_filter", self, label="Classes to display"))
        self.menu.append(ui.Info_Text("Comma-separated class names to display; empty = all."))
        self.classes_text = ui.Info_Text("Seen classes: -")
        self.menu.append(self.classes_text)
        self.menu.append(ui.Switch("show_masks", self, label="Draw segmentation masks"))
        self.menu.append(ui.Switch("show_ids", self, label="Show track ids"))
        self.menu.append(ui.Switch("sight_only", self, label="Only observed object"))
        self.menu.append(ui.Button("Reload objects data", self.load_objects_data))

        self.menu.append(ui.Separator())
        self.menu.append(ui.Info_Text("Offline reprocessing (overwrites objects.pldata):"))
        self.menu.append(ui.Text_Input("server_address", self, label="Detector address"))
        self.menu.append(ui.Slider("conf", self, min=0.05, max=0.95, step=0.05, label="Confidence"))
        self.menu.append(ui.Slider("smooth", self, min=0.0, max=0.95, step=0.05, label="Temporal smoothing"))
        self.menu.append(ui.Button("Reprocess recording", self.start_reprocess))
        self.menu.append(ui.Button("Cancel reprocess", self.cancel_reprocess))
        self.status_text = ui.Info_Text(self._status_line())
        self.menu.append(self.status_text)

    def deinit_ui(self):
        self.remove_menu()
        self.status_text = None
        self.classes_text = None

    # --- replay data ----------------------------------------------------------------------
    def load_objects_data(self):
        self._index = {}
        rec_dir = getattr(self.g_pool, "rec_dir", None)
        if load_pldata_file is None or not rec_dir:
            return
        try:
            pl = load_pldata_file(rec_dir, "objects")
        except Exception as exc:
            logger.debug("no objects.pldata: %s", exc)
            return
        for datum in pl.data:
            try:
                fi = datum["frame_index"]
            except (KeyError, TypeError):
                continue
            if fi is not None:
                self._index[int(fi)] = datum
        logger.info("loaded %d object frames", len(self._index))

    # --- per-frame replay -----------------------------------------------------------------
    def recent_events(self, events):
        frame = events.get("frame")
        if frame is None:
            return
        self.img = frame.img
        self.height, self.width = self.img.shape[:2]

        self._whitelist = frozenset(
            c.strip().lower() for c in self.classes_filter.split(",") if c.strip())

        datum = self._index.get(getattr(frame, "index", -1))
        if datum is not None:
            gaze = datum.get("gaze_2d")
            self.gaze_position_2d = list(gaze) if gaze else []
            detections = [dict(d) for d in datum.get("objects", [])]
            for det in detections:
                if det.get("kind") != "layer":
                    self._seen_classes.add(det.get("name", "?"))
            focus_idx = self._find_observed(detections)
            self._draw_overlay(detections, focus_idx)

        self._update_status()

    # --- shared overlay / matching (mirrors the Capture plugin) ---------------------------
    def _visible(self, name):
        return not self._whitelist or name.lower() in self._whitelist

    def _show_object(self, det):
        """Display filter: class whitelist AND confidence threshold (replay shows >= conf)."""
        return self._visible(det.get("name", "")) and det.get("conf", 1.0) >= self.conf

    def _find_observed(self, detections):
        if not self.gaze_position_2d or not detections:
            return -1
        gx, gy = self.gaze_position_2d
        best_idx, best_area = -1, None
        for i, det in enumerate(detections):
            if det.get("kind") == "layer" or not self._show_object(det):
                continue
            mask = det.get("mask")
            inside, area = False, None
            if mask is not None and len(mask) >= 3:
                contour = np.array(mask, dtype=np.int32).reshape(-1, 1, 2)
                if cv2.pointPolygonTest(contour, (float(gx), float(gy)), False) >= 0:
                    inside = True
                    area = abs(cv2.contourArea(contour))
            else:
                x1, y1, x2, y2 = det["box"]
                if x1 <= gx <= x2 and y1 <= gy <= y2:
                    inside = True
                    area = abs((x2 - x1) * (y2 - y1))
            if inside and (best_area is None or area < best_area):
                best_idx, best_area = i, area
        return best_idx

    def _draw_overlay(self, detections, focus_idx):
        img = self.img
        for det in detections:
            if det.get("kind") != "layer":
                continue
            mask = det.get("mask")
            if not (self.show_masks and mask is not None and len(mask) >= 3):
                continue
            color = LAYER_COLORS.get(det.get("name"), LAYER_DEFAULT_COLOR)
            contour = np.array(mask, dtype=np.int32).reshape(-1, 1, 2)
            overlay = img.copy()
            cv2.fillPoly(overlay, [contour], color)
            cv2.addWeighted(overlay, LAYER_ALPHA, img, 1.0 - LAYER_ALPHA, 0, img)
            cv2.polylines(img, [contour], True, color, 1)

        for i, det in enumerate(detections):
            if det.get("kind") == "layer" or not self._show_object(det):
                continue
            if self.sight_only and i != focus_idx:
                continue
            color = COLOR_OBSERVED if i == focus_idx else COLOR_OTHER
            x1, y1, x2, y2 = (int(v) for v in det["box"])
            mask = det.get("mask")
            if self.show_masks and mask is not None and len(mask) >= 3:
                contour = np.array(mask, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [contour], True, color, 2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = "{} [{:.0f}%]".format(det.get("name", "?"), det.get("conf", 0) * 100)
            if self.show_ids:
                tid = det.get("id")
                label = "#{} {}".format(tid if tid is not None else "-", label)
            cv2.putText(img, label, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- offline reprocessing -------------------------------------------------------------
    def start_reprocess(self):
        if self._reproc and self._reproc.get("running"):
            return
        if PLData_Writer is None:
            logger.error("file_methods.PLData_Writer unavailable")
            return
        self._reproc = {"running": True, "cancel": False, "done": 0, "total": 0, "msg": "starting..."}
        threading.Thread(target=self._reprocess_worker, daemon=True).start()

    def cancel_reprocess(self):
        if self._reproc and self._reproc.get("running"):
            self._reproc["cancel"] = True

    def _locate_world(self, rec_dir):
        for name in ("world.mp4", "world.mjpeg", "world.avi"):
            p = os.path.join(rec_dir, name)
            if os.path.exists(p):
                return p
        hits = glob.glob(os.path.join(rec_dir, "world*.mp4"))
        return hits[0] if hits else None

    def _load_gaze(self, rec_dir):
        """Return (timestamps ndarray, list of norm_pos) for recorded gaze, or (None, None)."""
        if load_pldata_file is None:
            return None, None
        try:
            pl = load_pldata_file(rec_dir, "gaze")
        except Exception:
            return None, None
        ts, norm = [], []
        for g in pl.data:
            try:
                norm.append(tuple(g["norm_pos"]))
                ts.append(float(g["timestamp"]))
            except (KeyError, TypeError):
                continue
        if not ts:
            return None, None
        return np.asarray(ts), norm

    def _reprocess_worker(self):
        st = self._reproc
        rec_dir = getattr(self.g_pool, "rec_dir", None)
        try:
            world = self._locate_world(rec_dir) if rec_dir else None
            ts_path = os.path.join(rec_dir, "world_timestamps.npy") if rec_dir else None
            if not world or not ts_path or not os.path.exists(ts_path):
                st.update(running=False, msg="world video / timestamps not found")
                return
            world_ts = np.load(ts_path)
            gaze_ts, gaze_norm = self._load_gaze(rec_dir)
            cap = cv2.VideoCapture(world)
            writer = PLData_Writer(rec_dir, "objects")

            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, 15000)
            sock.setsockopt(zmq.SNDTIMEO, 15000)
            sock.connect(self.server_address)

            total = len(world_ts)
            st["total"] = total
            for i in range(total):
                if st["cancel"]:
                    st["msg"] = "cancelled"
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                t = float(world_ts[i])
                gaze2d = self._gaze_at(gaze_ts, gaze_norm, t, w, h)

                try:
                    ok_enc, jpg = cv2.imencode(".jpg", frame)
                    sock.send(msgpack.packb({"jpg": jpg.tobytes(), "conf": self.conf,
                                             "smooth": self.smooth}, use_bin_type=True))
                    rep = msgpack.unpackb(sock.recv(), raw=False)
                except zmq.ZMQError as exc:
                    st.update(running=False, msg="detector not responding: {}".format(exc))
                    cap.release(); writer.close(); sock.close(0); ctx.term()
                    return
                detections = rep.get("detections", [])

                datum = self._build_datum(t, i, gaze2d, detections)
                writer.append(datum)
                st["done"] = i + 1
                st["msg"] = "{}/{} frames".format(i + 1, total)

            cap.release()
            writer.close()
            sock.close(0)
            ctx.term()
            if not st["cancel"]:
                st["msg"] = "done: {} frames".format(st["done"])
            st["running"] = False
            self.load_objects_data()
        except Exception as exc:  # pragma: no cover
            logger.exception("reprocess failed")
            st.update(running=False, msg="error: {}".format(exc))

    @staticmethod
    def _gaze_at(gaze_ts, gaze_norm, t, w, h, tol=0.05):
        if gaze_ts is None or len(gaze_ts) == 0:
            return None
        idx = int(np.searchsorted(gaze_ts, t))
        cands = [j for j in (idx - 1, idx) if 0 <= j < len(gaze_ts)]
        if not cands:
            return None
        j = min(cands, key=lambda k: abs(gaze_ts[k] - t))
        if abs(gaze_ts[j] - t) > tol:
            return None
        nx, ny = gaze_norm[j]
        return [nx * w, (1.0 - ny) * h]

    def _build_datum(self, timestamp, frame_index, gaze2d, detections):
        focus_idx = self._observed_index(detections, gaze2d)
        focus = None
        if focus_idx >= 0:
            d = detections[focus_idx]
            focus = {"id": d.get("id"), "name": d["name"], "conf": d["conf"],
                     "box": d["box"], "mask": d.get("mask")}
        return {
            "topic": "objects",
            "timestamp": float(timestamp),
            "frame_index": int(frame_index),
            "gaze_2d": list(gaze2d) if gaze2d else None,
            "focus": focus,
            # offline: keep full data (masks for every detection) for faithful replay
            "objects": [{"id": d.get("id"), "kind": d.get("kind", "object"), "engine": d.get("engine"),
                         "name": d["name"], "conf": d["conf"], "box": d["box"], "mask": d.get("mask")}
                        for d in detections],
        }

    @staticmethod
    def _observed_index(detections, gaze2d):
        """Pure version of _find_observed for the worker thread (no shared state, no filter)."""
        if not gaze2d or not detections:
            return -1
        gx, gy = gaze2d
        best_idx, best_area = -1, None
        for i, det in enumerate(detections):
            if det.get("kind") == "layer":
                continue
            mask = det.get("mask")
            inside, area = False, None
            if mask is not None and len(mask) >= 3:
                contour = np.array(mask, dtype=np.int32).reshape(-1, 1, 2)
                if cv2.pointPolygonTest(contour, (float(gx), float(gy)), False) >= 0:
                    inside = True
                    area = abs(cv2.contourArea(contour))
            else:
                x1, y1, x2, y2 = det["box"]
                if x1 <= gx <= x2 and y1 <= gy <= y2:
                    inside = True
                    area = abs((x2 - x1) * (y2 - y1))
            if inside and (best_area is None or area < best_area):
                best_idx, best_area = i, area
        return best_idx

    # --- status / display -----------------------------------------------------------------
    def _status_line(self):
        if self._reproc and self._reproc.get("running"):
            return "Reprocessing: " + self._reproc.get("msg", "")
        if self._reproc and self._reproc.get("msg"):
            return "Reprocess " + self._reproc["msg"]
        if self._index:
            return "Replay: {} object frames loaded".format(len(self._index))
        return "No objects.pldata — use Reprocess to generate it"

    def _update_status(self):
        if self.status_text is not None:
            self.status_text.text = self._status_line()
        if self.classes_text is not None:
            seen = sorted(self._seen_classes)
            shown = ", ".join(seen[:18]) + (" ..." if len(seen) > 18 else "")
            self.classes_text.text = "Seen classes: " + (shown if seen else "-")

    def gl_display(self):
        gl_utils.clear_gl_screen()
        gl_utils.make_coord_system_norm_based()
        try:
            draw_gl_texture(self.img)
            gl_utils.make_coord_system_pixel_based(self.img.shape)
        except (AttributeError, TypeError):
            pass

    def get_init_dict(self):
        return {
            "server_address": self.server_address,
            "conf": self.conf,
            "smooth": self.smooth,
            "sight_only": self.sight_only,
            "show_masks": self.show_masks,
            "show_ids": self.show_ids,
            "classes_filter": self.classes_filter,
        }

    def cleanup(self):
        self.cancel_reprocess()
