#!/usr/bin/env python3

import playground_detection
import ball_detection
import numpy as np
import cv2
import utilities
from utilities import read_metadata
from utilities import update_metadata

from matplotlib import pyplot as plt

# def imread(filename):
#     return cv2.imread("images/microsoft_cam/raw/" + filename)



def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

if __name__ == "__main__":
    img_path = "images/series-1/latest.png"
    img = cv2.imread(img_path)
    meta = read_metadata(img_path)

    if "playground_poly" not in meta:
        playground_poly = utilities.select_polygon(img)
        meta = update_metadata(img_path, {"playground_poly": playground_poly})

    playground_mask = utilities.poly2mask(meta["playground_poly"], img)

    # utilities.show(playground_mask)
    # utilities.show(cv2.bitwise_not(playground_mask))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,0]

    utilities.show(hue)
    utilities.show(playground_mask)

    plt.figure(1)
    plt.subplot(3,1,1)
    # Hue histogram playground
    utilities.plot_histogram(hue, [0], playground_mask, "b", max=256)
    # Hue histogram background
    utilities.plot_histogram(hue, [0], cv2.bitwise_not(playground_mask), "r", max=256)

    plt.subplot(3,1,2)
    h8 = (((hue/255.0)**8)*255.0).astype(np.uint8)

    # Hue histogram playground
    utilities.plot_histogram(h8, [0], playground_mask, "b", max=256)
    # Hue histogram background
    utilities.plot_histogram(h8, [0], cv2.bitwise_not(playground_mask), "r", max=256)


    # utilities.show(apply_mask(hue, cv2.bitwise_not(playground_mask)))

    plt.subplot(3,1,3)
    # Hue histogram of center box
    box_size = 60
    utilities.plot_histogram(
        utilities.box(hue, (int(hue.shape[0]/2), int(hue.shape [1]/2)), box_size),
        color="b", max=256)

    plt.show()
