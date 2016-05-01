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


def record_time_lapse():
    import datetime
    import utilities
    import image
    import time
    dir = "images/red_balls/series-1/"
    with NetworkCamera("http://31.45.53.135:1337/new_image.png") as cam:
        params = {
            "saturation" : [100, 150, 200],
            "exposure_absolute" : [5,10,20],
            "brightness" : [20, 40, 80],
        }
        def paramstring():
            p = []
            for key in params.keys():
                p.append("{}={}".format(key, str(cam.settings[key])))
            return ",".join(p)

        while True:
            import utilities
            for sat in params["saturation"]:
                for exp in params["exposure_absolute"]:
                    for bri in params["brightness"]:
                        begin = datetime.datetime.now()
                        print("capture start")
                        cam.settings["saturation"] = sat
                        cam.settings["exposure_absolute"] = exp
                        cam.settings["brightness"] = bri
                        frame = cam.capture()

                        ts_string = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                        filename = "{}-{}.png".format(ts_string, paramstring())
                        file_path = dir+filename
                        cv2.imwrite(file_path, frame)
                        # BUG: doesn't work if there's common metadata (metadata.json) in the dir
                        utilities.update_metadata(file_path, cam.settings)

                        end = datetime.datetime.now()
                        elapsed_s = (end-begin).total_seconds()

                        print("capture took:", elapsed_s)
                        time.sleep(max(0, 30-elapsed_s))



if __name__ == "__main__":
    with NetworkCamera("http://31.45.53.135:1337/raspberry_image.png") as cam:
        while True:
            import utilities
            frame = cam.capture()
            import image
            utilities.show_all(image.Image(image_data=frame), time_ms=0)
