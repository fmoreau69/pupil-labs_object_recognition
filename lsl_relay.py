#!/usr/bin/env python
"""LSL relay for the Pupil object-recognition plugin.

Subscribes to the plugin's data-export ZMQ PUB socket (enable "Stream object data (RTMaps/LSL)" in
the plugin) and re-publishes the object data as LSL streams, so it can be recorded and time-synced
with other sensors (gaze, vehicle, EEG, GSR...) via LabRecorder / pylsl. pylsl lives here in the
Python 3.12 venv rather than in the Pupil bundle (Python 3.6).

Two outlets, both timestamped with pylsl.local_clock():
  * "<name>"        — numeric (float32), 7 channels: [observed, x1, y1, x2, y2, gaze_x, gaze_y]
                      (observed = 1 if the participant is looking at an object, else 0)
  * "<name>_json"   — string marker stream carrying the full per-frame datum as JSON

Usage:
    python lsl_relay.py                                   # connect tcp://127.0.0.1:5561
    python lsl_relay.py --connect tcp://192.168.1.10:5561 --name PupilObjects
"""
import argparse
import json
import socket

import msgpack
import zmq
from pylsl import StreamInfo, StreamOutlet, local_clock, IRREGULAR_RATE


def parse_args():
    p = argparse.ArgumentParser(description="ZMQ->LSL relay for Pupil object data")
    p.add_argument("--connect", default="tcp://127.0.0.1:5561",
                   help="Plugin export socket address (use the Pupil host IP on a LAN)")
    p.add_argument("--name", default="PupilObjects", help="LSL stream name prefix")
    return p.parse_args()


def make_numeric_outlet(name, source_id):
    info = StreamInfo(name, "ObjectGaze", 7, IRREGULAR_RATE, "float32", source_id)
    chans = info.desc().append_child("channels")
    for label in ("observed", "box_x1", "box_y1", "box_x2", "box_y2", "gaze_x", "gaze_y"):
        chans.append_child("channel").append_child_value("label", label)
    return StreamOutlet(info)


def make_json_outlet(name, source_id):
    info = StreamInfo(name + "_json", "Markers", 1, IRREGULAR_RATE, "string", source_id + "_json")
    return StreamOutlet(info)


def main():
    args = parse_args()
    source_id = "pupil_objrec_" + socket.gethostname()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(args.connect)
    sub.setsockopt_string(zmq.SUBSCRIBE, "objects")
    print(f"[lsl_relay] subscribed to {args.connect}")

    numeric = make_numeric_outlet(args.name, source_id)
    markers = make_json_outlet(args.name, source_id)
    print(f"[lsl_relay] LSL outlets ready: '{args.name}' (float32x7) and '{args.name}_json' (string)")

    n = 0
    try:
        while True:
            _topic, payload = sub.recv_multipart()
            datum = msgpack.unpackb(payload, raw=False)
            ts = local_clock()

            focus = datum.get("focus") or {}
            box = focus.get("box") or [0.0, 0.0, 0.0, 0.0]
            gaze = datum.get("gaze_2d") or [0.0, 0.0]
            present = 1.0 if datum.get("focus") else 0.0
            numeric.push_sample(
                [present, float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                 float(gaze[0]), float(gaze[1])], ts)
            markers.push_sample([json.dumps(datum)], ts)

            n += 1
            if n % 100 == 0:
                print(f"[lsl_relay] relayed {n} frames")
    except KeyboardInterrupt:
        print("\n[lsl_relay] shutting down")
    finally:
        sub.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
