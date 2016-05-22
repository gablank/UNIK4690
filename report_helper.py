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

        def extract(img, c):
            return extract_bb(img, circle_bb(c, 5))

        image = Image(filename, histogram_equalization=None)
        ball = Image(image_data=extract(image.bgr,
                                        rs[0]),
                     histogram_equalization=None)

        ycrcb = ball.get_ycrcb(np.uint8)
        bgr = ball.get_bgr(np.uint8)
        hsv = ball.get_hsv(np.uint8)


        action(prefix+os.path.basename(filename),
               compose([bgr, bgr[:,:,2], ycrcb[:,:,1], hsv[:,:,0]][:3]))


def extract_multiple(filenames, basename):
    """
    Extract the same region from multiple images
    """
    img1 = cv2.imread(filenames[0])
    poly = utilities.select_polygon(img1)
    rect = cv2.boundingRect(np.array(poly))
    for i, filename in enumerate(filenames):
        img = cv2.imread(filename)
        cv2.imwrite(basename+str(i)+"-"+os.path.basename(filename),
                    utilities.extract_bb(img, rect))

if __name__ == '__main__':

    filenames = []

    if len(sys.argv) > 1:
        filenames = sys.argv[1:]

    extract_multiple(filenames[1:], filenames[0])
    exit(0)

    make_red_ball_imgs(filenames, write, prefix="report_imgs/all-")
    # make_red_ball_imgs(filenames, write, prefix="report_imgs/Cr-bgr-")
