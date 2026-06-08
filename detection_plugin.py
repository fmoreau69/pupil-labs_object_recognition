"""
Object Recognition (YOLO) — Pupil Capture plugin.

Lightweight plugin that runs inside the Pupil Capture *bundle* (Python 3.6). It does NOT run
YOLO itself: the heavy ultralytics/torch inference lives in an external process
(``yolo_server.py``, Python 3.12 venv) that the plugin talks to over a ZMQ REQ/REP socket on
localhost.

Responsibilities of this plugin:
  * grab the world frame + gaze in ``recent_events``;
  * hand the frame to a background worker thread that round-trips it to the detector
    (so the world process is never blocked by inference);
  * match the gaze position against detected boxes/masks to find the *observed* object;
  * draw the overlay (observed object in red, others in green);
  * publish object data on the Pupil IPC backbone (topic ``objects``);
  * write object data into the recording directory so it can be reloaded in Pupil Player.

Only dependencies already present in the bundle are used: pyzmq, numpy, opencv, msgpack, pyglui.
"""
import logging
import os
import queue
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
    from file_methods import PLData_Writer
except Exception:  # pragma: no cover - API guard
    PLData_Writer = None

logger = logging.getLogger(__name__)

# Colors are in BGR (OpenCV order).
COLOR_OBSERVED = (12, 15, 145)    # red  (#9C210C) — object the participant is looking at
COLOR_OTHER = (172, 183, 114)     # green/teal (#72B7AC) — other detected objects

# Semantic layers (road/lane/...) are drawn as translucent fills, by name.
LAYER_COLORS = {
    "drivable area": (180, 200, 90),
    "drivable area (sam3)": (180, 200, 90),
    "lane": (40, 200, 255),
}
LAYER_DEFAULT_COLOR = (200, 120, 200)
LAYER_ALPHA = 0.35


class DetectorClient(threading.Thread):
    """Background thread that ships frames to the external YOLO server and keeps the latest result.

    The plugin submits the most recent frame via :meth:`submit` (old frames are dropped) and reads
    the latest detections via :meth:`get_result`. The ZMQ socket lives entirely in this thread
    (ZMQ sockets are not thread-safe). On timeout/error the socket is recreated so a missing or
    restarted server never wedges the plugin.
    """

    def __init__(self, address, recv_timeout_ms=1000):
        super().__init__(daemon=True)
        self.address = address
        self.recv_timeout_ms = recv_timeout_ms

        self._inbox = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._result = None          # latest reply dict from the server
        self._stop = threading.Event()
        self.connected = False
        self.last_error = None
        self.infer_fps = 0.0

    # --- producer side (called from the world thread) -------------------------------------
    def submit(self, frame_bgr, params):
        """Queue the latest frame (+ inference params), dropping any previous un-processed one."""
        if self._inbox.full():
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                pass
        try:
            self._inbox.put_nowait((frame_bgr, params))
        except queue.Full:
            pass

    def get_result(self):
        with self._lock:
            return self._result

    def stop(self):
        self._stop.set()

    # --- worker thread --------------------------------------------------------------------
    def _new_socket(self, ctx):
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.recv_timeout_ms)
        sock.connect(self.address)
        return sock

    def run(self):
        ctx = zmq.Context()
        sock = self._new_socket(ctx)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                try:
                    frame, params = self._inbox.get(timeout=0.3)
                except queue.Empty:
                    continue

                t_start = time.time()
                ok, jpg = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                msg = {"jpg": jpg.tobytes()}
                msg.update(params)
                payload = msgpack.packb(msg, use_bin_type=True)

                try:
                    sock.send(payload)
                    socks = dict(poller.poll(self.recv_timeout_ms))
                    if socks.get(sock) == zmq.POLLIN:
                        reply = msgpack.unpackb(sock.recv(), raw=False)
                        with self._lock:
                            self._result = reply
                        self.connected = True
                        self.last_error = reply.get("error")
                        dt = time.time() - t_start
                        if dt > 0:
                            self.infer_fps = 0.7 * self.infer_fps + 0.3 * (1.0 / dt)
                    else:
                        raise zmq.ZMQError(msg="recv timeout")
                except zmq.ZMQError as exc:
                    # Reset the (now stuck) REQ socket and mark disconnected.
                    self.connected = False
                    self.last_error = str(exc)
                    poller.unregister(sock)
                    sock.close(0)
                    sock = self._new_socket(ctx)
                    poller.register(sock, zmq.POLLIN)
        finally:
            sock.close(0)
            ctx.term()


class Object_Recognition_YOLO(Plugin):
    """Detects/segments objects in the world view and flags the one the participant is looking at."""

    icon_chr = "O"

    def __init__(self, g_pool, server_address="tcp://127.0.0.1:5560", conf=0.25,
                 smooth=0.5, max_rate=30.0, sight_only=False, show_masks=True, show_ids=False,
                 classes_filter="", detect_floor=0.10):
        super().__init__(g_pool)
        self.order = 0.1  # run early so downstream plugins can use the "objects" events

        # settings (persisted via get_init_dict)
        self.server_address = server_address
        self.conf = conf                  # DISPLAY confidence threshold (filters overlay + focus)
        self.detect_floor = detect_floor  # detection floor sent to the server (data keeps everything above this)
        self.smooth = smooth              # temporal smoothing strength (0..0.95), applied server-side
        self.max_rate = max_rate          # max detections/s; overlay holds between inferences
        self.sight_only = sight_only      # only draw/export the observed object
        self.show_masks = show_masks
        self.show_ids = show_ids          # show track ids in labels
        # comma-separated class names to DISPLAY; empty = show all. The detector still finds
        # everything (full data is recorded); this only filters what is drawn / eligible as focus.
        self.classes_filter = classes_filter
        self.activation_state = False     # detection on/off

        # runtime state
        self._whitelist = frozenset()     # parsed, lowercased classes_filter
        self._seen_classes = set()        # object class names seen so far (for discovery)
        self.classes_text = None
        self.menu = None
        self.status_text = None
        self.img = None
        self.width, self.height = 1280, 720
        self.gaze_position_2d = []
        self.client = None
        self._writer = None
        self._draw_fps = 0.0
        self._last_submit = 0.0

    # --- UI -------------------------------------------------------------------------------
    def init_ui(self):
        self.add_menu()
        self.menu.label = "Object Recognition (YOLO)"
        self.menu.append(ui.Info_Text(
            "Detects and labels objects in the world camera. The object the participant is "
            "looking at is drawn in red, the others in green. Object data is published on the "
            "IPC backbone (topic \"objects\") and saved into the recording.\n"
            "Requires the external detector: run `python yolo_server.py` in the Python 3.12 venv."
        ))
        self.menu.append(ui.Text_Input("server_address", self, label="Detector address"))
        self.menu.append(ui.Slider("conf", self, min=0.05, max=0.95, step=0.05,
                                   label="Display confidence"))
        self.menu.append(ui.Info_Text(
            "Objects below this confidence are hidden from the overlay and ignored for the observed "
            "object — but the detector still finds them and ALL detections (down to the detection "
            "floor) are recorded, so nothing is lost."
        ))
        self.menu.append(ui.Slider("smooth", self, min=0.0, max=0.95, step=0.05,
                                   label="Temporal smoothing"))
        self.menu.append(ui.Info_Text(
            "Higher smoothing = steadier contours but a touch more lag. Requires tracking "
            "(on by default in the detector)."
        ))
        self.menu.append(ui.Slider("max_rate", self, min=1.0, max=30.0, step=1.0,
                                   label="Max detection rate (Hz)"))
        self.menu.append(ui.Info_Text(
            "Caps inference rate; the overlay holds the last result between detections. Lower it "
            "to steady the overlay or to save GPU on a weaker machine."
        ))
        self.menu.append(ui.Text_Input("classes_filter", self, label="Classes to display"))
        self.menu.append(ui.Info_Text(
            "Comma-separated class names to display (e.g. \"person, car, bus, traffic light\"). "
            "Empty = show all. The detector keeps finding everything and all detections are still "
            "recorded; this only filters the overlay and which objects can be the observed one."
        ))
        self.classes_text = ui.Info_Text("Seen classes: -")
        self.menu.append(self.classes_text)
        self.menu.append(ui.Switch("show_masks", self, label="Draw segmentation masks"))
        self.menu.append(ui.Switch("show_ids", self, label="Show track ids"))
        self.menu.append(ui.Switch("sight_only", self, label="Only observed object"))
        self.menu.append(ui.Button("Reconnect detector", self.restart_client))
        self.menu.append(ui.Switch("activation_state", self, label="Launch object recognition",
                                   setter=self._toggle_activation))
        self.status_text = ui.Info_Text("Detector: idle")
        self.menu.append(self.status_text)

    def deinit_ui(self):
        self.remove_menu()
        self.status_text = None
        self.classes_text = None

    def _toggle_activation(self, value):
        self.activation_state = value
        if value:
            self.start_client()
        else:
            self.stop_client()

    # --- detector client lifecycle --------------------------------------------------------
    def start_client(self):
        if self.client is None:
            self.client = DetectorClient(self.server_address)
            self.client.start()
            self._last_submit = 0.0

    def stop_client(self):
        if self.client is not None:
            self.client.stop()
            self.client = None

    def restart_client(self):
        self.stop_client()
        if self.activation_state:
            self.start_client()

    # --- recording hooks ------------------------------------------------------------------
    def on_notify(self, notification):
        subject = notification.get("subject", "")
        if subject == "recording.started":
            self._open_writer(notification.get("rec_path"))
        elif subject == "recording.stopped":
            self._close_writer()

    def _open_writer(self, rec_path):
        self._close_writer()
        if PLData_Writer is None or not rec_path:
            return
        try:
            self._writer = PLData_Writer(rec_path, "objects")
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not open PLData_Writer: %s", exc)
            self._writer = None

    def _close_writer(self):
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # pragma: no cover
                pass
            self._writer = None

    # --- per-frame processing -------------------------------------------------------------
    def recent_events(self, events):
        t_prev = time.time()
        events["objects"] = []

        frame = events.get("frame")
        if frame is None:
            return
        self.img = frame.img
        self.height, self.width = self.img.shape[:2]

        # Gaze in pixel coordinates (norm_pos has the y axis flipped vs. the image).
        gaze = events.get("gaze", [])
        try:
            norm = gaze[0]["norm_pos"]
            self.gaze_position_2d = [norm[0] * self.width, (1.0 - norm[1]) * self.height]
        except (IndexError, KeyError, TypeError):
            self.gaze_position_2d = []

        if not self.activation_state or self.client is None:
            self._update_status()
            return

        # Hand the current frame to the worker (throttled to max_rate) and overlay the latest
        # available result. Between submissions the held result keeps the overlay steady.
        now = time.time()
        min_period = 1.0 / self.max_rate if self.max_rate > 0 else 0.0
        if now - self._last_submit >= min_period:
            self.client.submit(self.img.copy(), {"conf": self.detect_floor, "smooth": self.smooth})
            self._last_submit = now
        result = self.client.get_result()
        detections = result.get("detections", []) if result else []

        # Parse the display whitelist and track which object classes we've seen (for discovery).
        self._whitelist = frozenset(
            c.strip().lower() for c in self.classes_filter.split(",") if c.strip())
        for det in detections:
            if det.get("kind") != "layer":
                self._seen_classes.add(det["name"])

        focus_idx = self._find_observed(detections)
        self._draw_overlay(detections, focus_idx)

        datum = self._build_datum(frame, detections, focus_idx)
        if datum is not None:
            events["objects"].append(datum)
            self._publish(datum)
            if self._writer is not None:
                try:
                    self._writer.append(datum)
                except Exception as exc:  # pragma: no cover
                    logger.debug("PLData append failed: %s", exc)

        dt = time.time() - t_prev
        if dt > 0:
            self._draw_fps = 0.8 * self._draw_fps + 0.2 * (1.0 / dt)
        self._update_status()

    def _visible(self, name):
        """Whether an object class is in the display whitelist (empty whitelist = all)."""
        return not self._whitelist or name.lower() in self._whitelist

    def _show_object(self, det):
        """Object passes the display filters: class whitelist AND confidence threshold.

        Display-only: every detection is still recorded (down to the detection floor); this just
        controls what is drawn and what can be the observed object.
        """
        return self._visible(det.get("name", "")) and det.get("conf", 1.0) >= self.conf

    def _find_observed(self, detections):
        """Return the index of the smallest *object* containing the gaze point, or -1.

        Semantic layers (road, lanes...) are skipped: the gaze is almost always on the road, so it
        would never leave it. Only instance objects are eligible to be "observed".
        """
        if not self.gaze_position_2d or not detections:
            return -1
        gx, gy = self.gaze_position_2d
        best_idx, best_area = -1, None
        for i, det in enumerate(detections):
            if det.get("kind") == "layer" or not self._show_object(det):
                continue
            mask = det.get("mask")
            inside = False
            area = None
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
        # Semantic layers first (translucent fills), so objects/labels draw on top.
        for det in detections:
            if det.get("kind") != "layer":
                continue
            mask = det.get("mask")
            if not (self.show_masks and mask is not None and len(mask) >= 3):
                continue
            color = LAYER_COLORS.get(det["name"], LAYER_DEFAULT_COLOR)
            contour = np.array(mask, dtype=np.int32).reshape(-1, 1, 2)
            overlay = img.copy()
            cv2.fillPoly(overlay, [contour], color)
            cv2.addWeighted(overlay, LAYER_ALPHA, img, 1.0 - LAYER_ALPHA, 0, img)
            cv2.polylines(img, [contour], True, color, 1)

        # Instance objects (gaze-coloured).
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
            label = "{} [{:.0f}%]".format(det["name"], det["conf"] * 100)
            if self.show_ids:
                # "#-" means the detector returned no track id (running without tracking?).
                tid = det.get("id")
                label = "#{} {}".format(tid if tid is not None else "-", label)
            cv2.putText(img, label, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _build_datum(self, frame, detections, focus_idx):
        if not detections:
            return None
        focus = None
        if focus_idx >= 0:
            d = detections[focus_idx]
            focus = {"id": d.get("id"), "name": d["name"], "conf": d["conf"],
                     "box": d["box"], "mask": d.get("mask")}
        return {
            "topic": "objects",
            "timestamp": frame.timestamp,
            "frame_index": frame.index,
            "gaze_2d": list(self.gaze_position_2d) if self.gaze_position_2d else None,
            "focus": focus,
            "objects": [{"id": d.get("id"), "kind": d.get("kind", "object"), "engine": d.get("engine"),
                         "name": d["name"], "conf": d["conf"], "box": d["box"]}
                        for d in detections],
        }

    def _publish(self, datum):
        ipc_pub = getattr(self.g_pool, "ipc_pub", None)
        if ipc_pub is None:
            return
        try:
            ipc_pub.send(datum)
        except Exception as exc:  # pragma: no cover - API guard
            logger.debug("ipc_pub.send failed: %s", exc)

    def _update_status(self):
        if self.classes_text is not None:
            seen = sorted(self._seen_classes)
            shown = ", ".join(seen[:18]) + (" ..." if len(seen) > 18 else "")
            self.classes_text.text = "Seen classes: " + (shown if seen else "-")
        if self.status_text is None:
            return
        if self.client is None:
            self.status_text.text = "Detector: idle"
            return
        if self.client.connected:
            self.status_text.text = "Detector: connected — infer {:.1f} Hz / plugin {:.1f} Hz".format(
                self.client.infer_fps, self._draw_fps)
        else:
            err = self.client.last_error or "no response"
            self.status_text.text = "Detector: DISCONNECTED ({}) — is yolo_server.py running?".format(err)

    # --- GL display -----------------------------------------------------------------------
    def gl_display(self):
        gl_utils.clear_gl_screen()
        gl_utils.make_coord_system_norm_based()
        try:
            draw_gl_texture(self.img)
            gl_utils.make_coord_system_pixel_based(self.img.shape)
        except (AttributeError, TypeError):
            pass

    # --- persistence / teardown -----------------------------------------------------------
    def get_init_dict(self):
        return {
            "server_address": self.server_address,
            "conf": self.conf,
            "smooth": self.smooth,
            "max_rate": self.max_rate,
            "detect_floor": self.detect_floor,
            "sight_only": self.sight_only,
            "show_masks": self.show_masks,
            "show_ids": self.show_ids,
            "classes_filter": self.classes_filter,
        }

    def cleanup(self):
        self.stop_client()
        self._close_writer()
