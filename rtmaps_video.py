"""RTMaps Python component: receive the annotated world video from the Pupil plugin over ZMQ.

Drop this into an RTMaps Python block. It SUBSCRIBEs to the plugin's annotated-video PUB socket
(enable "Stream annotated video (RTMaps)" in the plugin, default bind tcp://*:5562) and outputs the
overlaid world frames as an RTMaps image stream.

Plain pyzmq (the old imagezmq dependency is gone). Frames are JPEG, multipart [b"frame", jpg].
"""
import cv2
import numpy as np
import zmq

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
        self.add_property("sub_address", "tcp://127.0.0.1:5562")
        self.add_output("image", rtmaps.types.IPL_IMAGE)

    def Birth(self):
        addr = self.properties["sub_address"].data
        ctx = zmq.Context.instance()
        self.sub = ctx.socket(zmq.SUB)
        self.sub.connect(addr)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "frame")
        self.sub.set_hwm(2)
        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)
        rt.report_info("Subscribed to {}".format(addr))

    def Core(self):
        # Drain to the most recent frame so RTMaps stays in sync with the live stream.
        jpg = None
        while dict(self.poller.poll(5)).get(self.sub) == zmq.POLLIN:
            _topic, jpg = self.sub.recv_multipart()
        if jpg is None:
            return

        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        out = rtmaps.types.Ioelt()
        out.data = rtmaps.types.IplImage()
        out.data.color_model = "RGB"
        out.data.channel_seq = "RGB"
        out.data.image_data = np.ascontiguousarray(img)
        self.outputs["image"].write(out)

    def Death(self):
        if self.sub is not None:
            self.sub.close(0)
        rt.report_info("Death")
