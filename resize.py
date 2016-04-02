#!/usr/bin/python3
import numpy as np
import cv2 as cv
from os import walk


def resize_keep_aspect_ratio(img, new_size):
    new_w, new_h = new_size
    cur_h, cur_w = img.shape[:2]

    cur_aspect_ratio = cur_w / cur_h
    new_aspect_ratio = new_w / new_h

    pad_x, pad_y = 0, 0
    if cur_aspect_ratio < new_aspect_ratio:
        pad_x = int((cur_h * new_aspect_ratio - cur_w) / 2)
    elif cur_aspect_ratio > new_aspect_ratio:
        pad_y = int((cur_w / new_aspect_ratio - cur_h) / 2)

    # Pad original image so the aspect ratio equals the aspect ratio of the new resolution
    img = cv.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv.BORDER_CONSTANT, value=[0, 0, 0])

    resized = cv.resize(img, new_size, interpolation=cv.INTER_CUBIC)

    return resized

# Resizes all images found in images/raw into all other sizes used (currently only 1920x1080)
# Keeps the image aspect ratio by padding one dimension with a black border (if necessary)
if __name__ == "__main__":
    target_sizes = [(3840, 2160), (1920, 1080), (800, 600)]

    filenames = []
    for cur in walk("images/raw"):
        filenames = cur[2]
        break

    for filename in filenames:
        if filename is not "." and filename is not "..":
            img = cv.imread("images/raw/" + filename)

            for res_x, res_y in target_sizes:
                print("Resizing {} to {}x{}".format(filename, res_x, res_y))
                # resized = resize_keep_aspect_ratio(img, (res_x, res_y))
                # cv.imwrite("images/" + str(res_x) + "x" + str(res_y) + "/" + filename, resized)

            # Rescale:
            rescaled = cv.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
            cv.imwrite("images/0.5/" + filename, rescaled)
            rescaled = cv.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_CUBIC)
            cv.imwrite("images/0.25/" + filename, rescaled)

