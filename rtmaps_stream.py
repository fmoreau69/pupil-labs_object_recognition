# python zmq_consumer.py --port 5555
import rtmaps.types
import os
import cv2
import argparse
import numpy as np
import imagezmq
from rtmaps.base_component import BaseComponent  # base class
import rtmaps.core as rt
import rtmaps.reading_policy
from PIL import Image


# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self)  # call base class constructor
        self.force_reading_policy(rtmaps.reading_policy.SAMPLING)
        self.SERVER_PORT = 12001
        self.server_port = self.SERVER_PORT
        self.open_port = 'tcp://127.0.0.1:{}'.format(self.server_port)
        self.image_hub = imagezmq.ImageHub(open_port=self.open_port, REQ_REP=False)
        # Add properties
        #self.add_property("IP", '127.0.0.1')    # If you talk to a different machine use its IP.
        #self.add_property("Port", 5555)    # The port defaults to 5555.

    # All inputs, outputs and properties MUST be created in the
    # Dynamic() function.
    def Dynamic(self):
        self.add_output("images", rtmaps.types.IPL_IMAGE)  # define output

    # Birth() will be called once at diagram execution startup
    def Birth(self):
        print('Open Port is {}'.format(self.open_port))

    # Core() is called every time you have a new input
    def Core(self):
        print('Receiving frames...')
        out = rtmaps.types.Ioelt()
        out.data = rtmaps.types.IplImage()
        out.data.color_model = "COLR"  # Color model can also be : RGB, RGBA, YUV, YUVA, GRAY... (SDK C++ for more)
        out.data.channel_seq = "RGB"
        # show streamed images
        while True:
            # tpye(jpg_buffer) is <class 'zmq.sugar.frame.Frame'>
            host_name, jpg_buffer = self.image_hub.recv_jpg()
            #self.image_hub.send_reply(b'OK')
            # image is 1-d numpy.ndarray and decode to 3-d array
            image = np.frombuffer(jpg_buffer, dtype='uint8')
            image = cv2.imdecode(image, -1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            out.data.image_data = np.array(image)
            self.outputs["images"].write(out)  # and write it to the output

    # Death() will be called once at diagram execution shutdown
    def Death(self):
        self.image_hub.close()
        rt.report_info("Death")
