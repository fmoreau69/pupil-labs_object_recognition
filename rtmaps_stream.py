"""RTMaps Python component: receive object-recognition data from the Pupil plugin over ZMQ.

Drop this into an RTMaps Python block. It SUBSCRIBEs to the plugin's data-export PUB socket
(enable "Stream object data (RTMaps/LSL)" in the plugin, default bind tcp://*:5561) and exposes the
per-frame object data as RTMaps outputs.

Replaces the previous imagezmq-based version: plain pyzmq, no extra dependency.

Wire format (msgpack), multipart [b"objects", payload]:
    {topic, timestamp, frame_index, gaze_2d:[x,y]|None,
     focus:{id,name,conf,box:[x1,y1,x2,y2],mask}|None,
     objects:[{id,kind,engine,name,conf,box}, ...]}
"""
import json

import numpy as np
import zmq
import msgpack

import rtmaps.types
import rtmaps.core as rt
import rtmaps.reading_policy
from rtmaps.base_component import BaseComponent


class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self)
        self.force_reading_policy(rtmaps.reading_policy.SAMPLING)
        self.sub = None
        self.poller = None

    def Dynamic(self):
        # Address of the plugin's export socket. Use the Pupil host IP on a LAN.
        self.add_property("sub_address", "tcp://127.0.0.1:5561")
        # Outputs
        self.add_output("observed_name", rtmaps.types.TEXT_UTF8)     # observed object class, "" if none
        self.add_output("observed_box", rtmaps.types.FLOAT64)        # [x1, y1, x2, y2]
        self.add_output("observed_id", rtmaps.types.INTEGER64)       # track id, -1 if none
        self.add_output("gaze", rtmaps.types.FLOAT64)                # [x, y] in world pixels
        self.add_output("n_objects", rtmaps.types.INTEGER64)
        self.add_output("pupil_timestamp", rtmaps.types.FLOAT64)
        self.add_output("json", rtmaps.types.TEXT_UTF8)              # full datum as JSON

    def Birth(self):
        addr = self.properties["sub_address"].data
        ctx = zmq.Context.instance()
        self.sub = ctx.socket(zmq.SUB)
        self.sub.connect(addr)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "objects")
        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)
        rt.report_info("Subscribed to {}".format(addr))

    def Core(self):
        # Drain to the most recent message so RTMaps stays in sync with the live stream.
        datum = None
        while dict(self.poller.poll(5)).get(self.sub) == zmq.POLLIN:
            _topic, payload = self.sub.recv_multipart()
            datum = msgpack.unpackb(payload, raw=False)
        if datum is None:
            return

        focus = datum.get("focus") or {}
        box = focus.get("box") or [0.0, 0.0, 0.0, 0.0]
        gaze = datum.get("gaze_2d") or [0.0, 0.0]
        ts = float(datum.get("timestamp", 0.0))

        self._write("observed_name", focus.get("name", ""), ts)
        self._write("observed_box", [float(v) for v in box], ts)
        self._write("observed_id", int(focus["id"]) if focus.get("id") is not None else -1, ts)
        self._write("gaze", [float(gaze[0]), float(gaze[1])], ts)
        self._write("n_objects", int(len(datum.get("objects", []))), ts)
        self._write("pupil_timestamp", ts, ts)
        self._write("json", json.dumps(datum), ts)

    def _write(self, name, value, ts):
        out = rtmaps.types.Ioelt()
        out.data = value
        out.ts = int(ts * 1e6)  # RTMaps timestamps are in microseconds
        self.outputs[name].write(out)

    def Death(self):
        if self.sub is not None:
            self.sub.close(0)
        rt.report_info("Death")
