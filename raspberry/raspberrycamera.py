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


"""
Available settings:
https://picamera.readthedocs.io/en/release-1.10/api_camera.html
"""


class RaspberryCamera(camera.Camera):
    def __init__(self):
        self._camera_device = "Raspberry Pi camera"

        self._settings = {}

        self._get_settings()

        super(RaspberryCamera, self).__init__()

    def capture(self):
        stream = io.BytesIO()
        with picamera.PiCamera() as cam:
            # Apply settings
            cam.resolution = (self._frame_width, self._frame_height)
            for setting, value in self._settings.items():
                setattr(cam, setting, value)

            cam.start_preview()
            # Let the camera warm up
            time.sleep(2)
            cam.capture(stream, format="png")
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        bgr = cv2.imdecode(data, 1)
        return bgr

    def set(self, property, value):
        self._settings[property] = value

    def get(self, property):
        if property in self._settings:
            return self._settings[property]
        return None

    def set_resolution(self, width, height):
        self._frame_width = int(width)
        self._frame_height = int(height)

    def get_resolution(self):
        return self._frame_width, self._frame_height

    def _get_settings(self):
        with picamera.PiCamera() as cam:
            self._frame_width, self._frame_height = cam.resolution
            self._settings["contrast"] = cam.contrast
            self._settings["sharpness"] = cam.sharpness
            self._settings["brightness"] = cam.brightness
            self._settings["zoom"] = cam.zoom
            self._settings["shutter_speed"] = cam.shutter_speed
            self._settings["saturation"] = cam.saturation
            self._settings["exposure_compensation"] = cam.exposure_compensation

    def __str__(self):
        as_str = "Camera read from: {}\n".format(self._camera_device)
        as_str += "Camera settings:\n"
        as_str += "Resolution: {}x{}\n".format(self._frame_width, self._frame_height)
        for setting, value in self._settings.items():
            as_str += setting.capitalize() + ": {}\n".format(value)
        return as_str

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
