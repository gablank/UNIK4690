#!/usr/bin/python3
import cv2
import numpy as np
import utilities
import os


class Image:
    def __init__(self, path=None, image_data=None, color_normalization=True):
        if (path is None and image_data is None) or (path is not None and image_data is not None):
            raise RuntimeError("One and only one of path and image_data may be not None!")

        self.filename = None
        if path is not None:
            self.path = utilities.locate_file(path)

            self.bgr = cv2.imread(path)
            self.filename = os.path.basename(path)
            if self.bgr is None:
                raise FileNotFoundError("Unable to load image from {}".format(path))
        else:
            self.path = None

        if image_data is not None:
            self.bgr = image_data

        if color_normalization:
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
        self.color_space_dict = None

    def get_metadata(self):
        if self.path is None:
            return {}
        return utilities.read_metadata(self.path)

    def get_bgr(self, dtype=np.float32):
        return utilities.astype(self.bgr, dtype)

    def get_gray(self, dtype=np.float32):
        img = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        return utilities.astype(img, dtype)

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

    # Get the image space dict with image dtype=np.float32
    def get_color_space_dict(self):
        if self.color_space_dict is None:
            self.color_space_dict = {}
            bgr = self.get_bgr(np.float32)
            hsv = self.get_hsv(np.float32)
            lab = self.get_lab(np.float32)
            ycrcb = self.get_ycrcb(np.float32)
            self.color_space_dict["bgr_b"] = bgr[:, :, 0]
            self.color_space_dict["bgr_g"] = bgr[:, :, 1]
            self.color_space_dict["bgr_r"] = bgr[:, :, 2]
            self.color_space_dict["hsv_h"] = hsv[:, :, 0]
            self.color_space_dict["hsv_s"] = hsv[:, :, 1]
            self.color_space_dict["hsv_v"] = hsv[:, :, 2]
            self.color_space_dict["lab_l"] = lab[:, :, 0]
            self.color_space_dict["lab_a"] = lab[:, :, 1]
            self.color_space_dict["lab_b"] = lab[:, :, 2]
            self.color_space_dict["ycrcb_y"] = ycrcb[:, :, 0]
            self.color_space_dict["ycrcb_cr"] = ycrcb[:, :, 1]
            self.color_space_dict["ycrcb_cn"] = ycrcb[:, :, 2]
        return self.color_space_dict

    def get_light_mask(self):
        """
        Get a mask where 1's means that the pixel is in an area that are saturated by sunlight
        """
        raise NotImplementedError()


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
