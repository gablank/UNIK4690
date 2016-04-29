#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This file must be compatible with Python 2 and 3!
"""

BRIGHTNESS, \
CONTRAST, \
SATURATION, \
WHITE_BALANCE_TEMPERATURE_AUTO, \
WHITE_BALANCE_TEMPERATURE, \
SHARPNESS, \
BACKLIGHT_COMPENSATION, \
EXPOSURE_AUTO, \
EXPOSURE, \
FOCUS_AUTO, \
FOCUS, \
ZOOM \
    = range(12)


class Camera(object):
    def __init__(self):
        self._frame_width = None
        self._frame_height = None

    def capture(self):
        raise NotImplementedError

    def set_defaults(self):
        self.set_resolution(1920, 1080)
        self.set(BRIGHTNESS, 30)
        self.set(CONTRAST, 5)
        self.set(SATURATION, 200)
        self.set(WHITE_BALANCE_TEMPERATURE_AUTO, 0)
        self.set(WHITE_BALANCE_TEMPERATURE, 5000)
        self.set(SHARPNESS, 50)
        self.set(BACKLIGHT_COMPENSATION, 0)
        self.set(EXPOSURE_AUTO, 1)  # Not a bool, but mapping. 1 means manual, 3 is auto
        self.set(EXPOSURE, 5)
        self.set(FOCUS_AUTO, 0)
        self.set(FOCUS, 2)
        self.set(ZOOM, 0)

    def set(self, property, value):
        raise NotImplementedError

    def get(self, property):
        raise NotImplementedError

    def set_resolution(self, width, height):
        raise NotImplementedError

    def get_resolution(self):
        raise NotImplementedError

    def _property_to_string(self, cv2_property):
        return {
            BRIGHTNESS: ("brightness", 30, 255),
            CONTRAST: ("contrast", 0, 10),
            SATURATION: ("saturation", 0, 200),
            WHITE_BALANCE_TEMPERATURE_AUTO: ("white_balance_temperature_auto", 0, 1),
            WHITE_BALANCE_TEMPERATURE: ("white_balance_temperature", 2500, 10000),
            SHARPNESS: ("sharpness", 0, 50),
            BACKLIGHT_COMPENSATION: ("backlight_compensation", 0, 10),
            EXPOSURE_AUTO: ("exposure_auto", 1, 3),
            EXPOSURE: ("exposure_absolute", 1, 10000),
            FOCUS_AUTO: ("focus_auto", 0, 1),
            FOCUS: ("focus_absolute", 0, 40),
            ZOOM: ("zoom_absolute", 0, 317),
        }[cv2_property]

    def __str__(self):
        as_str = "Camera read from: {}\n".format(self._camera_device)
        as_str += "Camera settings:\n"
        as_str += "Resolution: {}x{}\n".format(self._frame_width, self._frame_height)
        as_str += "Brightness: {}\n".format(self.get(BRIGHTNESS))
        as_str += "Contrast: {}\n".format(self.get(CONTRAST))
        as_str += "Saturation: {}\n".format(self.get(SATURATION))
        as_str += "Auto white balance: {}\n".format(self.get(WHITE_BALANCE_TEMPERATURE_AUTO))
        as_str += "White balance temperature: {}\n".format(self.get(WHITE_BALANCE_TEMPERATURE))
        as_str += "Sharpness: {}\n".format(self.get(SHARPNESS))
        as_str += "Backlight compensation: {}\n".format(self.get(BACKLIGHT_COMPENSATION))
        as_str += "Auto exposure: {}\n".format(self.get(EXPOSURE_AUTO))
        as_str += "Exposure: {}\n".format(self.get(EXPOSURE))
        as_str += "Auto focus: {}\n".format(self.get(FOCUS_AUTO))
        as_str += "Focus: {}\n".format(self.get(FOCUS))
        as_str += "Zoom: {}\n".format(self.get(ZOOM))

        return as_str
