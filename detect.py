#!/usr/bin/python3
import playground_detection
import ball_detection
import numpy as np
import cv2
from os import walk


def detection_method(img):
    S = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg = np.average(S)
    ret, S = cv2.threshold(S, 2*avg, 255, cv2.THRESH_BINARY)
    cv2.imshow("S", S)
    cv2.waitKey(0)


if __name__ == "__main__":
    read_from_camera = False

    if read_from_camera:
        cap = cv2.VideoCapture(1)

        while True:
            ret, frame = cap.read()
            playing_field = playground_detection.detect(frame, "flood_fill")
            #ball_detection.show_keypoints(playing_field)
            #cv2.imshow("output", playing_field)

    else:
        filenames = []
        for cur in walk("images/raw/"):
            filenames = cur[2]
            break

        for filename in filenames:
            if filename is not "." and filename is not "..":
                # img = cv2.imread("images/1920x1080/" + filename)
                img = cv2.imread("images/0.25/" + filename)
                # img = cv2.imread("images/800x600/" + filename)
                # img = cv2.imread("images/3840x2160/" + filename)
                # showKeypoints(img)
                playing_field = playground_detection.detect(img, "flood_fill", draw_field=True)
                cv2.waitKey(0)
                #ball_detection.show_keypoints(playing_field)
