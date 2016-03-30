#!/usr/bin/python3
import numpy as np
import matplotlib as plt
import cv2 as cv
from os import walk


def showKeypoints(img):
    # works well
    # surf = cv.xfeatures2d.SURF_create(700)
    # img = cv.GaussianBlur(img, (17, 17), 0)

    # works very well
    surf = cv.xfeatures2d.SURF_create(1900)
    blurred = cv.GaussianBlur(img, (7, 7), 0)

    # surf = cv.xfeatures2d.SURF_create(2500)
    # blurred = img

    # blurred = cv.GaussianBlur(img, (3, 3), 0)
    #  surf = cv.xfeatures2d.SIFT_create(80)

    kp, des = surf.detectAndCompute(blurred, None)

    keypointsImg = img.copy()
    cv.drawKeypoints(img, kp, keypointsImg, color=[255, 0, 0])

    cv.imshow("Keypoints", keypointsImg)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    filenames = []
    for cur in walk("images/raw/"):
        filenames = cur[2]
        break

    for filename in filenames:
        if filename is not "." and filename is not "..":
            img = cv.imread("images/1920x1080/" + filename)
            # img = cv.imread("images/800x600/" + filename)
            # img = cv.imread("images/3840x2160/" + filename)
            showKeypoints(img)
