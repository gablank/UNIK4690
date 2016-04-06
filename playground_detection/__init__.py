#!/usr/bin/python3
import cv2
import numpy as np
import playground_detection.maximize_density as maximize_density
import playground_detection.flood_fill as flood_fill


def detect(img, method="flood_fill", draw_field=False):
    polygon = [(0, 0), (len(img)-1, 0), (len(img)-1, len(img[0])-1), (0, len(img[0])-1)]
    if method == "flood_fill":
        polygon = flood_fill.detect(img)
    elif method == "maximize_density":
        polygon = maximize_density.detect(img)

    # return img
    g_b = img[:,:,0].copy()
    g_b[:,:] = 0
    g_b = cv2.fillConvexPoly(g_b, polygon, color=255)

    playing_field = img.copy()

    if draw_field:
        for idx in range(len(polygon)):
            if method == "maximize_density":
                pt1 = (polygon[idx][1], polygon[idx][0])
                pt2 = (polygon[(idx+1)%len(polygon)][1], polygon[(idx+1)%len(polygon)][0])
            else:
                pt1 = polygon[idx][0]
                pt2 = polygon[(idx+1)%len(polygon)][0]
                pt1 = (pt1[0], pt1[1])
                pt2 = (pt2[0], pt2[1])
            cv2.line(img, pt1, pt2, (0,0,255), 3)

        cv2.imshow("Playing field", img)
        cv2.waitKey(int(1000/30))
        # cv2.destroyAllWindows()

    # Use g_b as a "mask" for img (I couldn't figure out the proper way to use it as a mask)
    for i in (0, 1, 2):
        playing_field[:,:,i] = np.minimum(g_b, playing_field[:,:,i])

    return playing_field
