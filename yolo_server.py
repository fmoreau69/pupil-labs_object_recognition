#!/usr/bin/env python
"""External inference server for the Pupil Capture object-recognition plugin.

Runs in its own Python 3.12 venv (CUDA torch + ultralytics) and talks to the lightweight in-bundle
plugin over a ZMQ REQ/REP socket on localhost. The bundle's Python 3.6 cannot run torch, so all the
heavy inference lives here.

Engines (see engines.py), runnable alone or combined with --engines:
  * yolo     — detection/segmentation + ByteTrack tracking (real-time)
  * yolopv2  — drivable-area + lane-line semantic segmentation (real-time, needs yolopv2.pt)
  * sam3     — text-prompt segmentation (offline-grade, needs `sam3` package + HF token)

All engines feed a shared temporal smoother (per-id for instances, per-name for semantic layers)
so contours stop flickering frame to frame. Smoothing strength is set live by the plugin.

Wire protocol (msgpack, one request -> one reply):
    request : {"jpg": <bytes>, "conf": <float, optional>, "smooth": <float 0..0.95, optional>}
    reply   : {"detections": [<normalized detection dict>, ...],
               "shape": [h, w], "engines": [str, ...], "task": str}

Usage:
    python yolo_server.py                                   # yolo seg (default)
    python yolo_server.py --engines yolo,yolopv2            # objects + road/lane layers
    python yolo_server.py --engines yolopv2 --yolopv2-model path/to/yolopv2.pt
    python yolo_server.py --engines sam3 --sam3-road "drivable road surface in front of the vehicle"
"""
import argparse
import time

import cv2
import msgpack
import numpy as np
import zmq

from engines import TemporalSmoother, build_engines


def parse_args():
    p = argparse.ArgumentParser(description="Inference server for the Pupil object-recognition plugin")
    p.add_argument("--engines", default="yolo",
                   help="Comma-separated engines to run: yolo,yolopv2,sam3")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5560)
    p.add_argument("--conf", type=float, default=0.25, help="Default confidence threshold (yolo)")
    p.add_argument("--device", default=None, help="'0' (GPU), 'cpu' (default: auto)")
    p.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    # yolo
    p.add_argument("--model", default="yolo11n-seg.pt", help="Ultralytics model (yolo engine)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--mask-epsilon", type=float, default=1.5)
    p.add_argument("--no-track", action="store_true", help="Disable ByteTrack (yolo engine)")
    p.add_argument("--tracker", default="bytetrack.yaml")
    # yolopv2
    p.add_argument("--yolopv2-model", default="yolopv2.pt", help="YOLOPv2 TorchScript path")
    # sam3
    p.add_argument("--sam3-prompts", default="",
                   help="';'-separated text prompts for SAM3 markings")
    p.add_argument("--sam3-road", default="", help="SAM3 road-area prompt (empty = off)")
    # smoothing
    p.add_argument("--smooth", type=float, default=0.5,
                   help="Default temporal smoothing 0..0.95 (overridden per-request by the plugin)")
    p.add_argument("--smooth-scale", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    engines = build_engines(args)
    engine_names = [e.name for e in engines]
    task = next((e.task for e in engines if getattr(e, "name", "") == "yolo"), "segment")
    smoother = TemporalSmoother(scale=args.smooth_scale)
    print(f"[server] engines={engine_names} device={args.device or 'auto'} half={args.half}")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    endpoint = f"tcp://{args.host}:{args.port}"
    sock.bind(endpoint)
    print(f"[server] listening on {endpoint} — Ctrl+C to stop")

    n, t0 = 0, time.time()
    try:
        while True:
            raw = sock.recv()
            try:
                req = msgpack.unpackb(raw, raw=False)
                img = cv2.imdecode(np.frombuffer(req["jpg"], dtype=np.uint8), cv2.IMREAD_COLOR)
                conf = float(req.get("conf", args.conf))
                smooth_alpha = float(req.get("smooth", args.smooth))

                detections = []
                for engine in engines:
                    try:
                        detections.extend(engine.infer(img, conf))
                    except Exception as exc:
                        print(f"[server] engine {engine.name} error: {type(exc).__name__}: {exc}")
                detections = smoother.smooth(detections, img.shape, smooth_alpha)
                reply = {"detections": detections, "shape": [int(img.shape[0]), int(img.shape[1])],
                         "engines": engine_names, "task": task}
            except Exception as exc:  # never leave a REQ peer hanging
                reply = {"detections": [], "error": f"{type(exc).__name__}: {exc}"}
                print(f"[server] error: {reply['error']}")

            sock.send(msgpack.packb(reply, use_bin_type=True))

            n += 1
            if time.time() - t0 >= 5.0:
                print(f"[server] {n / (time.time() - t0):.1f} req/s, "
                      f"{len(reply.get('detections', []))} obj last frame")
                n, t0 = 0, time.time()
    except KeyboardInterrupt:
        print("\n[server] shutting down")
    finally:
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
