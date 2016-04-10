#!/usr/bin/python3
import cv2
import numpy as np
import utilities


def detect(img):
    middle = (int(img.shape[1] / 2), int(img.shape[0] / 2))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)

    hue = hsv[:,:,0] / np.amax(hsv[:,:,0])

    #hue = cv2.blur(hue, (3,3))
    hue = hue**16

    hue = hue * 255
    hue = hue.astype(np.uint8)

    #utilities._show(hue)
    iterations = 2
    size = 15

    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    hue = cv2.dilate(hue, dilationElement, iterations=iterations)

    utilities.show(hue, "after dilation")
    hue[np.where(hue == 255)] = 254
    box_size = 30
    hue[middle[1]-box_size:middle[1]+box_size, middle[0]-box_size:middle[0]+box_size] = 0
    mask = utilities.flood_fill_until(hue, 0.2)
    # diff = 2
    # num_filled = 0
    # while num_filled < 0.2*(len(img) * len(img[0])):
    #     num_filled, mask, _, _ = cv2.floodFill(hue.copy(), None, middle, 255, upDiff=diff, loDiff=diff, flags=cv2.FLOODFILL_FIXED_RANGE)
    #     diff += 1
    # utilities.show(mask, "mask")

    # hue[np.where(mask != 255)] = 0
    # utilities.show(hue, "hue")
    # return

    # hue *= 255
    # hue = hue.astype(np.uint8)
    # ret, hue = cv2.threshold(hue, 1, 255, cv2.THRESH_BINARY)
    #
    # utilities.show(hue)

    # im2, contours, hierarchy = cv2.findContours(hue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        polygon = contours[0]
        for i in range(1, len(contours)):
            polygon = np.concatenate((polygon, contours[i]))

        convex_hull = cv2.convexHull(polygon)
        # temp = utilities.draw_convex_hull(img, convex_hull)
        # utilities.show(temp, "contours")

        convex_hull = cv2.approxPolyDP(convex_hull, 5, True)
        # temp = utilities.draw_convex_hull(img, convex_hull)
        # utilities.show(temp, "contours")

        new_convex_hull = []
        for i in range(len(convex_hull)):
            if utilities.get_angle(convex_hull[(i-1)%len(convex_hull)][0], convex_hull[i][0], convex_hull[(i+1)%len(convex_hull)][0]) > 25:
                new_convex_hull.append(convex_hull[i])
        convex_hull = np.array(new_convex_hull)

        return convex_hull
