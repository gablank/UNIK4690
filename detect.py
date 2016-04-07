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
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        video = True
        ctr = 0
        while True:
            if video:
                ret, frame = cap.read()

                cv2.imwrite("images/microsoft_cam/video/" + str(ctr) + ".jpg", frame)
                cv2.imshow("written", frame)
                key = cv2.waitKey(int(1000/10))

                if key != -1:
                    break
                ctr += 1

            else:
                ret, frame = cap.read()

                cv2.imshow("written", frame)
                key = cv2.waitKey(0)

                # space
                if key == 1048608:
                    cv2.imwrite("images/microsoft_cam/raw/" + str(ctr) + ".jpg", frame)
                    ctr += 1

                # q
                if key == 1048689:
                    break

                _, _ = cap.read()
                _, _ = cap.read()
                _, _ = cap.read()
                _, _ = cap.read()

            #playing_field = playground_detection.detect(frame, "flood_fill", True)
            #ball_detection.show_keypoints(playing_field)
            #cv2.imshow("output", playing_field)

    else:
        #for i in range(385):
        #    img = cv2.imread("images/microsoft_cam/video/" + str(i) + ".jpg")
            #cv2.imshow("test", img)

        #    playing_field = playground_detection.detect(img, "flood_fill", draw_field=True)
        #    cv2.waitKey(16)
        #exit(0)

        filenames = []
        for cur in walk("images/microsoft_cam/raw/"):
        #for cur in walk("images/microsoft_cam/video/"):
        #for cur in walk("images/0.25/"):
            filenames = cur[2]
            break

        for filename in filenames:
            if filename is not "." and filename is not "..":
                # img = cv2.imread("images/1920x1080/" + filename)
                #img = cv2.imread("images/0.25/" + filename)
                img = cv2.imread("images/microsoft_cam/raw/" + filename)
                #img = cv2.imread("images/microsoft_cam/video/" + filename)
                #img = cv2.resize(img, (1024,768), interpolation=cv2.INTER_CUBIC)
                # img = cv2.imread("images/800x600/" + filename)
                # img = cv2.imread("images/3840x2160/" + filename)
                # showKeypoints(img)
                playing_field = playground_detection.detect(img, "flood_fill", draw_field=True)
                cv2.waitKey(0)
                #ball_detection.show_keypoints(playing_field)
