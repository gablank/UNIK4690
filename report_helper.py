#!/usr/bin/env python3

import sys
import cv2
import numpy as np
import utilities
from image import Image
from utilities import show_all, extract_bb, circle_bb
import os


def compose(images, border=2):
    w = sum([img.shape[1]+border for img in images]) - border
    h = max([img.shape[0] for img in images])
    canvas = np.zeros((h,w,3), dtype=np.uint8)
    x = 0
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h,w = img.shape[:2]
        canvas[0:h,x:x+w] = img

        x+=w+border

    return canvas

def show(source_filename, img):
    utilities.show(img)

def write(source_filename, img):
    cv2.imwrite(source_filename, img)

def make_red_ball_imgs(filenames, action=show, prefix=""):
    for filename in filenames:
        metadata = utilities.read_metadata(filename)
        rs = metadata["red_ball_circles"]

        image = Image(filename, histogram_equalization=None)

        ycrcb = image.get_ycrcb(np.uint8)
        bgr = image.get_bgr(np.uint8)

        def extract(img):
            return extract_bb(img, circle_bb(rs[0], 5))

        action(prefix+os.path.basename(filename),
               compose(list(map(extract, [bgr, ycrcb[:,:,1]]))))

if __name__ == '__main__':
    
    filenames = []

    if len(sys.argv) > 1:
        filenames = sys.argv[1:]

    make_red_ball_imgs(filenames, write, prefix="report_imgs/")
