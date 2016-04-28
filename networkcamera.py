#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This file must be compatible with Python 2 and 3!
"""
import sys
import camera
import tempfile
import cv2
if sys.version_info.major == 3:
    import urllib.request as urllib
else:
    import urllib


class NetworkCamera(camera.Camera):
    def __init__(self, url):
        self.url = url

        self._camera_device = self.url

        self.settings = {}

        super(NetworkCamera, self).__init__()

    def capture(self):
        url = self.url
        first = True
        for property in self.settings:
            if first:
                url += "?"
                first = False
            else:
                url += "&"
            url += property + "=" + str(self.settings[property])

        request = urllib.urlopen(url)
        pic_data = request.read()
        pic_file = tempfile.NamedTemporaryFile(suffix="png")
        pic_file.write(pic_data)
        return cv2.imread(pic_file.name)

    def set(self, property, value):
        property_string = property

        if isinstance(property, int):
            property_string, low, high = self._property_to_string(property)

            if low > value or high < value:
                raise ValueError("Value is outside of property range: {} <= {} <= {}".format(low, property_string, high))

        self.settings[property_string] = value

    def get(self, property):
        if property not in self.settings:
            return None
        return self.settings[property]

    def set_resolution(self, width, height):
        self.settings["width"] = width
        self.settings["height"] = height

    def get_resolution(self):
        return self.settings["width"], self.settings["height"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == "__main__":
    with NetworkCamera("http://31.45.53.135:1337/new_image.png") as cam:

        while True:
            import utilities
            frame = cam.capture()

            utilities.show(frame, time_ms=30, draw_histograms=True)
