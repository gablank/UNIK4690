#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This file must be compatible with Python 2 and 3!
"""
import camera
import cv2
import picamera
import time
import io
import numpy as np


class RaspberryCamera(camera.Camera):
    def __init__(self):
        self._camera_device = "Raspberry Pi camera"

        self._frame_width = None
        self._frame_height = None

        super(RaspberryCamera, self).__init__()

    def capture(self):
        stream = io.BytesIO()
        with picamera.PiCamera() as cam:
            cam.resolution = (self._frame_width, self._frame_height)
            cam.start_preview()
            # Let the camera warm up
            time.sleep(2)
            cam.capture(stream, format="png")
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        bgr = cv2.imdecode(data, 1)
        return bgr

    def set(self, property, value):
        pass

    def get(self, property):
        return 0

    def set_resolution(self, width, height):
        self._frame_width = width
        self._frame_height = height

    def get_resolution(self):
        return self._frame_width, self._frame_height

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == "__main__":
    with RaspberryCamera() as camera:

        while True:
            import utilities
            frame = camera.capture()

            utilities.show(frame, time_ms=30, draw_histograms=True)
