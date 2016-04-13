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

        video = False
        ctr = 0
        while True:
            ret, frame = cap.read()
            playing_field = playground_detection.detect(frame, "flood_fill", True)
            # ball_detection.show_keypoints(playing_field)
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("test", playing_field)
            key = cv2.waitKey(30)
            # q
            if key == 1048689:
                break
            continue

            if video:
                cv2.imwrite("images/microsoft_cam/video/" + str(ctr) + ".jpg", frame)
                cv2.imshow("written", frame)
                key = cv2.waitKey(int(1000/10))

                if key != -1:
                    break
                ctr += 1

            else:
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

    else:
        for cur in walk("images/microsoft_cam/24h/south/"):
            filenames = cur[2]
            break
        print(filenames)

        filenames.sort()

        for file in filenames:
            img = cv2.imread("images/microsoft_cam/24h/south/" + file)
            cv2.putText(img, file, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,2]
            #img = img / np.amax(img)
            #cv2.imshow("test", img)
            #cv2.waitKey(30)
            playground_detection.detect(img, "flood_fill", draw_field=True)
           #playing_field = playground_detection.detect(img, "flood_fill", draw_field=True)
           #cv2.waitKey(30)
        exit(0)

        filenames = []
        for cur in walk("images/microsoft_cam/raw/"):
        # for cur in walk("images/microsoft_cam/video/"):
        # for cur in walk("images/raw/"):
            filenames = cur[2]
            break

        for filename in filenames:
            if filename is not "." and filename is not "..":
                # img = cv2.imread("images/1920x1080/" + filename)
                # img = cv2.imread("images/0.25/" + filename)
                img = cv2.imread("images/microsoft_cam/raw/" + filename)
                #img = cv2.imread("images/microsoft_cam/video/" + filename)
                #img = cv2.resize(img, (1024,768), interpolation=cv2.INTER_CUBIC)
                # img = cv2.imread("images/800x600/" + filename)
                # img = cv2.imread("images/3840x2160/" + filename)
                # showKeypoints(img)
                # img = cv2.imread("images/raw/" + filename)
                playing_field = playground_detection.detect(img, "flood_fill", draw_field=True)
                #cv2.waitKey(0)
                #ball_detection.show_keypoints(playing_field)
