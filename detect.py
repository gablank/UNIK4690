#!/usr/bin/python3
import numpy as np
import matplotlib as plt
import cv2
from os import walk



def showKeypoints(img):
    # works well
    # surf = cv.xfeatures2d.SURF_create(700)
    # img = cv.GaussianBlur(img, (17, 17), 0)

    # works very well
    # surf = cv.xfeatures2d.SURF_create(3000)
    #blurred = img
    # grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # cv2.namedWindow('image')
    #
    # def cannyAndShow(x):
    #     lowerLimit = cv2.getTrackbarPos("Lower", "image")
    #     upperLimit = cv2.getTrackbarPos("Upper", "image")
    #     blur = cv2.getTrackbarPos("Blur", "image")
    #     if blur % 2 != 1:
    #         blur += 1
    #         cv2.setTrackbarPos("Blur", "image", blur)
    #     blurred = cv2.GaussianBlur(grayscale, (blur, blur), 0)
    #     lowerLimit = upperLimit/2
    #     cv2.setTrackbarPos("Lower", "image", int(lowerLimit))
    #     edges = cv2.Canny(blurred, lowerLimit, upperLimit)
    #     cv2.imshow("image", edges)
    #
    # # create trackbars for color change
    # cv2.createTrackbar('Blur', 'image', 0, 13, cannyAndShow)
    # cv2.createTrackbar('Lower', 'image', 0, 1000, cannyAndShow)
    # cv2.createTrackbar('Upper', 'image', 0, 1000, cannyAndShow)
    #
    # cannyAndShow(None)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # circles = cv2.HoughCircles(grayscale, cv2.HOUGH_GRADIENT, 1, 20, param1=upperLimit, param2=20, minRadius=5, maxRadius=22)
    #
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0,:]:
    #         # draw the outer circle
    #         cv2.circle(grayscale,(i[0],i[1]),i[2],(0,255,0),2)
    #         # draw the center of the circle
    #         cv2.circle(grayscale,(i[0],i[1]),2,(0,0,255),3)
    #
    #     cv2.imshow("test", grayscale)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return
    surf = cv2.xfeatures2d.SURF_create(2500)
    blurred = img

    LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = LAB[:,:,0]
    A = LAB[:,:,1]
    B = LAB[:,:,2]
    #cv2.imshow("L", L)
    #cv2.waitKey(0)
    #cv2.imshow("A", A)
    #cv2.waitKey(0)
    #cv2.imshow("B", B)
    #cv2.waitKey(0)

    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = HSV[:,:,0]
    S = HSV[:,:,1]
    V = HSV[:,:,2]
    #cv2.imshow("H", H)
    #cv2.waitKey(0)
    #cv2.imshow("S", S)
    #cv2.waitKey(0)
    #cv2.imshow("V", V)
    #cv2.waitKey(0)

    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:,:,0]
    Cr = YCrCb[:,:,1]
    Cv = YCrCb[:,:,2]
    #cv2.imshow("Y", Y)
    #cv2.waitKey(0)
    #cv2.imshow("Cr", Cr)
    #cv2.waitKey(0)
    #cv2.imshow("Cv", Cv)
    #cv2.waitKey(0)

    #Y = cv2.GaussianBlur(Y, (1, 1), 0)
    #cv2.imshow("y", Y)
    #cv2.waitKey(0)
    Y_float = Y.astype(np.float32)
    Y_float *= 1./255

    Y2 = np.power(Y_float, 2)
    #cv2.imshow("Y_float", Y2)
    #cv2.waitKey(0)
    Y3 = np.power(Y_float, 3)
    #cv2.imshow("Y_float", Y3)
    #cv2.waitKey(0)
    Y4 = np.power(Y_float, 4)
    #cv2.imshow("Y_float", Y4)
    #cv2.waitKey(0)
    # grayscale = img
    # grayscale = cv2.split(grayscale)[0]
    blurred = cv2.GaussianBlur(Y, (11, 11), 0)
    #  surf = cv2.xfeatures2d.SIFT_create(80)

    darkBlobParams = cv2.SimpleBlobDetector_Params()
    darkBlobParams.filterByArea = True
    darkBlobParams.minArea = 70
    darkBlobParams.maxArea = 150
    darkBlobParams.minDistBetweenBlobs = 1
    darkBlobParams.blobColor = 0
    darkBlobParams.filterByConvexity = False
    darkBlobDetector = cv2.SimpleBlobDetector_create(darkBlobParams)

    lightBlobParams = cv2.SimpleBlobDetector_Params()
    lightBlobParams.filterByArea = True
    lightBlobParams.minArea = 70
    lightBlobParams.maxArea = 500
    lightBlobParams.minDistBetweenBlobs = 1
    lightBlobParams.blobColor = 255
    lightBlobParams.filterByConvexity = False
    lightBlobDetector = cv2.SimpleBlobDetector_create(lightBlobParams)

    Y = Y3

    amax = np.amax(Y)
    light = Y / amax
    light *= 255
    light = np.clip(light, 0, 255)
    light = light.astype(np.uint8)

    avg = np.average(Y)
    dark = Y / (3*avg)
    dark *= 255
    dark = np.clip(dark, 0, 255)
    dark = dark.astype(np.uint8)

    dark = cv2.GaussianBlur(dark, (15, 15), 0)
    kpDark = darkBlobDetector.detect(dark)

    light = cv2.GaussianBlur(light, (11, 11), 0)
    kpLight = lightBlobDetector.detect(light)
    #kp, des = surf.detectAndCompute(blurred, None)

    keypointsImg = img.copy()
    cv2.drawKeypoints(img, kpDark, keypointsImg, color=[0, 0, 255])
    cv2.drawKeypoints(keypointsImg, kpLight, keypointsImg, color=[255, 0, 0])

    cv2.imshow("Keypoints", keypointsImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    filenames = []
    for cur in walk("images/raw/"):
        filenames = cur[2]
        break

    for filename in filenames:
        if filename is not "." and filename is not "..":
            img = cv2.imread("images/1920x1080/" + filename)
            # img = cv2.imread("images/800x600/" + filename)
            # img = cv2.imread("images/3840x2160/" + filename)
            showKeypoints(img)
