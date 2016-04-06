#!/usr/bin/python3
import cv2
import numpy as np


def _show(img, win_name="test"):
    return
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect(img):
    b, g, r = img[:,:,0].astype(int), img[:,:,1].astype(int), img[:,:,2].astype(int)

    # g_b = np.absolute(g - b)
    # r_b = np.absolute(r - b)
    r_g = np.absolute(r - g)

    # g_b = (g_b * (255 / np.amax(g_b))).astype(np.uint8)
    # r_b = (r_b * (255 / np.amax(r_b))).astype(np.uint8)
    r_g = (r_g * (255 / np.amax(r_g))).astype(np.uint8)

    # _show(g_b, "g_b")
    # _show(r_b, "r_b")
    _show(r_g, "abs(r-g)")

    # r_g = cv2.blur(r_g, (5,5))
    r_g = cv2.GaussianBlur(r_g, (9,9), 0)
    _, r_g = cv2.threshold(r_g, 30, 255, cv2.THRESH_BINARY)

    _show(r_g, "after blur and threshold")

    # For some reason, some images have the field in a lighter color than the rest
    # so if the field has a higher average value than the rest of the picture, we invert the picture
    middle = (int(img.shape[1] / 2), int(img.shape[1] / 2))
    box_size = 30
    avg = np.average(r_g[middle[0]-box_size:middle[0]+box_size, middle[1]-box_size:middle[1]+box_size])
    total_avg = np.average(r_g)
    if avg > total_avg:
        r_g = 255 - r_g
        avg = 255 - avg

        # _show(r_g)

    # _, r_g = cv2.threshold(r_g, 1.2*avg, 255, cv2.THRESH_BINARY)

    # _show(r_g, "after inversion")

    iterations = 5
    size = 5
    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    r_g = cv2.dilate(r_g, dilationElement, iterations=iterations)

    _show(r_g, "after dilation")

    erosionElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    r_g = cv2.erode(r_g, erosionElement, iterations=iterations)

    _show(r_g, "after erosion")

    middle = (int(img.shape[1] / 2), int(img.shape[1] / 2))
    result = cv2.floodFill(r_g, None, middle, 120)
    _, r_g, _, _ = result

    _show(r_g)

    r_g[np.where(r_g != 120)] = 0
    r_g[np.where(r_g != 0)] = 255

    _show(r_g)

    im2, contours, hierarchy = cv2.findContours(r_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(r_g, contours, 0, (255,255,0))
    # cv2.imshow("test", r_g)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    polygon = contours[0]
    # polygon = cv2.approxPolyDP(contours[0], 20, True)
    # polygon = cv2.approxPolyDP(polygon, 100, True)

    # return polygon
    return cv2.convexHull(polygon)
    return
    _, r_g = cv2.threshold(r_g, 30, 255, cv2.THRESH_BINARY)

    _show(r_g)

    middle = (int(img.shape[1] / 2), int(img.shape[1] / 2))

    box_size = 30
    avg = np.average(r_g[middle[0]-box_size:middle[0]+box_size, middle[1]-box_size:middle[1]+box_size])
    total_avg = np.average(r_g)
    if avg > total_avg:
        r_g = 255 - r_g
        avg = np.average(r_g[middle[0]-box_size:middle[0]+box_size, middle[1]-box_size:middle[1]+box_size])

    r_g = cv2.blur(r_g, (15,15), 0)

    _show(r_g)

    ret, r_g = cv2.threshold(r_g, 0.6*avg, 255, cv2.THRESH_BINARY)

    _show(r_g)

    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    r_g = cv2.dilate(r_g, dilationElement, iterations=5)

    _show(r_g)

    erosionElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    r_g = cv2.erode(r_g, erosionElement, iterations=10)

    _show(r_g)

    r_g = cv2.GaussianBlur(r_g, (25, 25), 0)

    _show(r_g)

    r_g = cv2.adaptiveThreshold(r_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 73, 1)

    _show(r_g)

    r_g = cv2.dilate(r_g, dilationElement, iterations=10)

    _show(r_g)

    middle = (int(img.shape[1] / 2), int(img.shape[1] / 2))
    result = cv2.floodFill(r_g, None, middle, 120)
    _, r_g, _, _ = result

    _show(r_g)

    r_g[np.where(r_g != 120)] = 0
    r_g[np.where(r_g != 0)] = 255

    _show(r_g)

    erosionSize = 7
    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosionSize, erosionSize))
    r_g = cv2.dilate(r_g, dilationElement, iterations=10)

    _show(r_g)

    erosionElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosionSize, erosionSize))
    r_g = cv2.erode(r_g, erosionElement, iterations=10)

    _show(r_g)

    im2, contours, hierarchy = cv2.findContours(r_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    polygon = contours[0]
    #polygon = cv2.approxPolyDP(contours[0], 50, True)
    #polygon = cv2.approxPolyDP(polygon, 100, True)

    return cv2.convexHull(polygon)
