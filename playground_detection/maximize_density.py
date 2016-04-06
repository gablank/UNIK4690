#!/usr/bin/python3
import cv2
import numpy as np


# lower score is better
def _calc_density_score(img, points):
    poly = np.zeros(img.shape)
    cv2.fillConvexPoly(poly, np.array(points), [255, 255, 255])

    area = np.count_nonzero(poly)
    poly = img.copy()
    cv2.fillConvexPoly(poly, np.array(points), [255, 255, 255])

    n_black_px = np.count_nonzero(poly) - np.count_nonzero(img)

    # Works fairly well
    # black_density = n_black_px / area
    #
    # return black_density - 0.0000001*area

    tot_black_density = (img.shape[0]*img.shape[1] - np.count_nonzero(img)) / (img.shape[0]*img.shape[1])

    black_density = n_black_px / area

    return black_density / (tot_black_density + 0.00000001*area)


def _calc_black_density(img, points):
    poly = np.zeros(img.shape)
    cv2.fillConvexPoly(poly, np.array(points), [255, 255, 255])

    area = np.count_nonzero(poly)
    poly = img.copy()
    cv2.fillConvexPoly(poly, np.array(points), [255, 255, 255])

    n_black_px = np.count_nonzero(poly) - np.count_nonzero(img)

    black_density = n_black_px / area

    return black_density


def _maximize_density(img):
    height, width = img.shape

    center_y = int(height / 2)
    center_x = int(width / 2)

    min_vertical_box = (center_x-1, center_x+1)
    max_vertical_box = (0, width-1)
    min_horizontal_box = (center_y-1, center_y+1)
    max_horizontal_box = (0, height-1)

    # Default to 4% of the minimum dimension
    initial_box_side = int(min(width, height) * 0.04)

    p0 = (center_x - initial_box_side, center_y - initial_box_side)
    p1 = (center_x - initial_box_side, center_y + initial_box_side)
    p2 = (center_x + initial_box_side, center_y + initial_box_side)
    p3 = (center_x + initial_box_side, center_y - initial_box_side)
    # return [p0, p1, p2, p3]
    # Polygon:
    #            p3
    #   p0
    #
    #  p1      p2
    #

    cur_score = _calc_density_score(img, [p0, p1, p2, p3])

    avg_num_black_px = (width*height - np.count_nonzero(img)) / (width*height)
    print("Num black pxs:", width*height - np.count_nonzero(img), "avg:", avg_num_black_px)
    threshold = avg_num_black_px / 0.8

    while True:
        change = False
        new_p0 = (max(0, p0[0]-10), p0[1])
        new_p1 = (max(0, p1[0]-10), p1[1])
        if new_p0 != p0 or new_p1 != p1:
            delta_score = _calc_black_density(img, [new_p0, new_p1, p1, p0])
            if delta_score < threshold:
                p0 = new_p0
                p1 = new_p1
                change = True

        new_p2 = (min(width-1, p2[0]+10), p2[1])
        new_p3 = (min(width-1, p3[0]+10), p3[1])
        if new_p2 != p2 or new_p3 != p3:
            delta_score = _calc_black_density(img, [p2, p3, new_p3, new_p2])
            if delta_score < threshold:
                p2 = new_p2
                p3 = new_p3
                change = True

        if not change:
            break

    while True:
        change = False
        new_p0 = (p0[0], max(0, p0[1]-10))
        new_p3 = (p3[0], max(0, p3[1]-10))
        if new_p0 != p0 or new_p3 != p3:
            delta_score = _calc_black_density(img, [new_p0, p0, p3, new_p3])
            if delta_score < threshold:
                p0 = new_p0
                p3 = new_p3
                change = True

        new_p1 = (p1[0], min(height-1, p1[1]+10))
        new_p2 = (p2[0], min(height-1, p2[1]+10))
        if new_p1 != p1 or new_p2 != p2:
            delta_score = _calc_black_density(img, [p1, new_p1, new_p2, p2])
            if delta_score < threshold:
                p1 = new_p1
                p2 = new_p2
                change = True

        if not change:
            break

    return [p0, p1, p2, p3]
    cur_score = _calc_density_score(img, [p0, p1, p2, p3])
    print(cur_score)
    rate = 2**6
    while rate > 0:
        change = False
        # p0
        p0s = []
        for di in (-rate, 0, rate):
            for dj in (-rate, 0, rate):
                if di == dj == 0:
                    continue

                new_p0 = (min(width-1, max(0, p0[0]+rate)), min(height-1, max(0, p0[1]+rate)))
                if new_p0 == p0:
                    continue
                if new_p0[0] + 100 > p3[0] or new_p0[1] + 100 > p1[1]:
                    continue

                new_score = _calc_density_score(img, [new_p0, p1, p2, p3])
                if new_score < cur_score:
                    p0s.append((new_p0, new_score))

        if len(p0s) > 0:
            change = True
            p0s.sort(key=lambda x: x[0])
            p0 = p0s[0][0]


        # p1
        p1s = []
        for di in (-rate, 0, rate):
            for dj in (-rate, 0, rate):
                if di == dj == 0:
                    continue

                new_p1 = (min(width-1, max(0, p1[0]+rate)), min(height-1, max(0, p1[1]+rate)))
                if new_p1 == p1:
                    continue
                if new_p1[0] + 100 > p2[0] or new_p1[1] - 100 < p0[1]:
                    continue

                new_score = _calc_density_score(img, [p0, new_p1, p2, p3])
                if new_score < cur_score:
                    p1s.append((new_p1, new_score))

        if len(p1s) > 0:
            change = True
            p1s.sort(key=lambda x: x[0])
            p1 = p1s[0][0]

        # p2
        p2s = []
        for di in (-rate, 0, rate):
            for dj in (-rate, 0, rate):
                if di == dj == 0:
                    continue

                new_p2 = (min(width-1, max(0, p2[0]+rate)), min(height-1, max(0, p2[1]+rate)))
                if new_p2 == p2:
                    continue
                if new_p2[0] - 100 < p1[0] or new_p2[1] - 100 < p3[1]:
                    continue

                new_score = _calc_density_score(img, [p0, p1, new_p2, p3])
                if new_score < cur_score:
                    p2s.append((new_p2, new_score))

        if len(p2s) > 0:
            change = True
            p2s.sort(key=lambda x: x[0])
            p2 = p2s[0][0]

        # p3
        p3s = []
        for di in (-rate, 0, rate):
            for dj in (-rate, 0, rate):
                if di == dj == 0:
                    continue

                new_p3 = (min(width-1, max(0, p3[0]+rate)), min(height-1, max(0, p3[1]+rate)))
                if new_p3 == p3:
                    continue
                if new_p3[0] - 100 < p0[0] or new_p3[1] + 100 > p2[1]:
                    continue

                new_score = _calc_density_score(img, [p0, p1, p2, new_p3])
                if new_score < cur_score:
                    p3s.append((new_p3, new_score))

        if len(p3s) > 0:
            change = True
            p3s.sort(key=lambda x: x[0])
            p3 = p3s[0][0]

        if not change:
            rate = int(rate / 2)
            print(rate)


    return [p0, p1, p2, p3]

    while True:
        change = False
        # p0
        new_p = (max(0, p0[0]-10), p0[1])
        new_score = _calc_density_score(img, [new_p, p1, p2, p3])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p0 = new_p
            cur_score = new_score
            change = True

        new_p = (p0[0], max(0, p0[1]-10))
        new_score = _calc_density_score(img, [new_p, p1, p2, p3])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p0 = new_p
            cur_score = new_score
            change = True

        # p1
        new_p = (max(0, p1[0]-10), p1[1])
        new_score = _calc_density_score(img, [p0, new_p, p2, p3])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p1 = new_p
            cur_score = new_score
            change = True

        new_p = (p1[0], min(height-1, p1[1]+10))
        new_score = _calc_density_score(img, [p0, new_p, p2, p3])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p1 = new_p
            cur_score = new_score
            change = True

        # p2
        new_p = (min(width-1, p2[0]+10), p2[1])
        new_score = _calc_density_score(img, [p0, p1, new_p, p3])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p2 = new_p
            cur_score = new_score
            change = True

        new_p = (p2[0], min(height-1, p2[1]+10))
        new_score = _calc_density_score(img, [p0, p1, new_p, p3])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p2 = new_p
            cur_score = new_score
            change = True

        # p3
        new_p = (min(width-1, p3[0]+10), p3[1])
        new_score = _calc_density_score(img, [p0, p1, p2, new_p])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p3 = new_p
            cur_score = new_score
            change = True

        new_p = (p3[0], min(0, p3[1]-10))
        new_score = _calc_density_score(img, [p0, p1, p2, new_p])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p3 = new_p
            cur_score = new_score
            change = True

        if not change:
            break

    return [p0, p1, p2, p3]


def detect(img):
    r, g, b = img[:,:,0].astype(int), img[:,:,1].astype(int), img[:,:,2].astype(int)
    r_g = np.absolute(r - g)
    r_b = np.absolute(r - b)
    g_b = np.absolute(g - b)

    g_b = (g_b * (255 / np.amax(g_b))).astype(np.uint8)

    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    erosionElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    g_b = cv2.dilate(g_b, dilationElement, iterations=4)
    g_b = cv2.erode(g_b, erosionElement, iterations=4)

    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    erosionElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    g_b = cv2.dilate(g_b, dilationElement, iterations=5)
    g_b = cv2.erode(g_b, erosionElement, iterations=10)

    g_b = cv2.GaussianBlur(g_b, (25, 25), 0)

    g_b = cv2.adaptiveThreshold(g_b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 73, 1)
    g_b = cv2.dilate(g_b, dilationElement, iterations=10)

    erosionSize = 7
    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosionSize, erosionSize))
    erosionElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosionSize, erosionSize))
    g_b = cv2.dilate(g_b, dilationElement, iterations=10)
    g_b = cv2.erode(g_b, erosionElement, iterations=10)

    return np.array(_maximize_density(g_b))
