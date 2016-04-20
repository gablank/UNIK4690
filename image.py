#!/usr/bin/python3
import cv2
import numpy as np
import utilities
import os


class Image:
    def __init__(self, path=None, image_data=None):
        if (path is None and image_data is None) or (path is not None and image_data is not None):
            raise RuntimeError("One and only one of path and image_data may be not None!")

        self.filename = None
        if path is not None:
            path = utilities.locate_file(path)

            self.bgr = cv2.imread(path)
            self.filename = os.path.basename(path)
            if self.bgr is None:
                raise FileNotFoundError("Unable to load image from {}".format(path))

        if image_data is not None:
            self.bgr = image_data

        # utilities.show(self.bgr)
        self.bgr = utilities.as_float32(self.bgr)
        target_averages = [0.636815, 0.543933, 0.469305]
        for i in range(3):
            self.bgr[:,:,i] += target_averages[i] - np.average(self.bgr[:,:,i])
        self.bgr[np.where(self.bgr > 1.0)] = 1.0
        self.bgr[np.where(self.bgr < 0.0)] = 0.0
        self.bgr = utilities.as_uint8(self.bgr)
        # utilities.show(self.bgr)

        self.hsv = None
        self.lab = None
        self.ycrcb = None

    def get_bgr(self, dtype=np.float32):
        return utilities.astype(self.bgr, dtype)

    def get_hsv(self, dtype=np.float32):
        if self.hsv is None:
            self.hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)
        return utilities.astype(self.hsv, dtype)

    def get_lab(self, dtype=np.float32):
        if self.lab is None:
            self.lab = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2LAB)
        return utilities.astype(self.lab, dtype)

    def get_ycrcb(self, dtype=np.float32):
        if self.ycrcb is None:
            self.ycrcb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2YCrCb)
        return utilities.astype(self.ycrcb, dtype)

    def get_light_mask(self):
        """
        Get a mask where 1's means that the pixel is in an area that are saturated by sunlight
        """
        pass


if __name__ == "__main__":
    img = Image("microsoft_cam/24h/south/2016-04-12_19:21:04.png")
    img = Image("2016-04-12_19:21:04.png")

    import os
    filenames = []
    for cur in os.walk(os.path.join(utilities.get_project_directory(), "images/microsoft_cam/24h/south/")):
        filenames = cur[2]
        break

    filenames.sort()

    for file in filenames:
        try:
            img = Image(os.path.join(utilities.get_project_directory(), "images/microsoft_cam/24h/south/", file))
        except FileNotFoundError:
            continue

        img.get_light_mask()
