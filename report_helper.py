#!/usr/bin/env python3

import sys
import cv2
import numpy as np
import utilities
from image import Image
from utilities import show_all, extract_bb, circle_bb, extract_circle, distance
import os
from matplotlib import pyplot as plt


def compose(images, border=2):
    w = sum([img.shape[1]+border for img in images]) - border
    h = max([img.shape[0] for img in images])
    canvas = np.ones((h,w,3), dtype=np.uint8)*255
    x = 0
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h,w = img.shape[:2]
        canvas[0:h,x:x+w] = img

        x+=w+border

    return canvas

def show(source_filename, img):
    print(source_filename)
    utilities.show(img)

def write(source_filename, img):
    cv2.imwrite(source_filename, img)

def make_red_ball_imgs(filenames, action=show, prefix=""):
    for filename in filenames:
        metadata = utilities.read_metadata(filename)
        balls = metadata["red_ball_circles"]

        image = Image(filename, histogram_equalization=None)
        ball = Image(image_data=extract_circle(image.bgr, balls[0], 5),
                     histogram_equalization=None)

        ycrcb = ball.get_ycrcb(np.uint8)
        bgr = ball.get_bgr(np.uint8)
        hsv = ball.get_hsv(np.uint8)

        action(prefix+os.path.basename(filename),
               compose([bgr, bgr[:,:,2], ycrcb[:,:,1], hsv[:,:,0]][:3]))

def ball_generator(filenames, margin=5, ball_idxs=[0]):
    for filename in filenames:
        metadata = utilities.read_metadata(filename)
        balls = metadata["ball_circles"]
        balls = sorted(balls, key=lambda c: c[1]) #key=lambda x: distance(x[0], (1920/2, 0)))

        image = Image(filename, histogram_equalization=None)

        if ball_idxs is None:
            ball_idxs = range(len(balls))

        ball_imgs = []

        for i in ball_idxs:
            ball_img = Image(image_data=extract_circle(image.bgr, balls[i], margin),
                             histogram_equalization=None)
            ball_imgs.append(ball_img)

        yield ball_imgs, filename


def make_balls_imgs(filenames, margin=5, action=show, prefix=""):
    for balls, filename in ball_generator(filenames, margin=margin, ball_idxs=None):
        action(prefix+os.path.basename(filename),
               compose([b.bgr for b in balls]))

def make_ball_imgs(filenames, margin=5, action=show, prefix=""):
    for (ball,), filename in ball_generator(filenames, margin=margin, ball_idxs=[0]):

        ycrcb = ball.get_ycrcb(np.uint8)
        bgr = ball.get_bgr(np.uint8)
        hsv = ball.get_hsv(np.uint8)

        action("", bgr)
        continue

        action(prefix+os.path.basename(filename),
               compose([bgr, bgr[:,:,2], ycrcb[:,:,1], hsv[:,:,0]][0]))

# def histogram():
    # plt.figure(1)
    # plt.subplot(3,1,1)
    # # Hue histogram playground
    # utilities.plot_histogram(hue, [0], playground_mask, "b", max=256)
    # # Hue histogram background
    # utilities.plot_histogram(hue, [0], cv2.bitwise_not(playground_mask), "r", max=256)


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

    # extract_multiple(filenames[1:], filenames[0])
    # exit(0)

    filenames = sorted(filenames)

    make_balls_imgs(filenames, margin=20, action=write, prefix="ri/all-balls/")
    # make_ball_imgs(filenames, margin=10, action=show, prefix="report_imgs/all-")
    # make_red_ball_imgs(filenames, action=write, prefix="report_imgs/all-")
    # make_red_ball_imgs(filenames, action=write, prefix="report_imgs/Cr-bgr-")
