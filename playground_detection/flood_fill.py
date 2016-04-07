#!/usr/bin/python3
import cv2
import numpy as np


def _show(img, win_name="test"):
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def detect(img):
    middle = (int(img.shape[1] / 2), int(img.shape[0] / 2))

    def middle_avg(img, count_blacks=False):
        x = int(len(img[0]) / 2)
        y = int(len(img) / 2)
        block = 50
        tot = np.sum(img[y - block:y + block, x - block:x + block])
        if count_blacks:
            num = img[y - block:y + block, x - block:x + block].size
        else:
            num = np.count_nonzero(img[y - block:y + block, x - block:x + block])
        return tot / num

    def threshold(img, band_width=30):
        avg = middle_avg(img)
        img[np.where(img < avg - band_width)] = 0
        img[np.where(img > avg + band_width)] = 0
        img[np.where(img > 0)] = 255
        return img

    def fill_threshold(img):
        import random
        random_size = 30
        img[np.where(img == 255)] = 254
        seed = middle
        num_filled, img, _, _ = cv2.floodFill(img, None, seed, 255)
        for i in range(-random_size, random_size):
            for j in range(-random_size, random_size):
                seed = (middle[0] + i, middle[1] + j)
                num_filled, img, _, _ = cv2.floodFill(img, None, seed, 255)
                if num_filled > 0.1*img.size:
                    break

        img[np.where(img < 255)] = 0


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)

    hue = hsv[:,:,0] / np.amax(hsv[:,:,0])

    #hue = cv2.blur(hue, (3,3))
    hue = hue**16

    hue = hue * 255
    hue = hue.astype(np.uint8)

    #_show(hue)
    iterations = 2
    size = 15

    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    hue = cv2.dilate(hue, dilationElement, iterations=iterations)

    #_show(hue, "after dilation")
    hue[np.where(hue == 255)] = 254
    diff = 2
    num_filled = 0
    box_size = 30
    hue[middle[1]-box_size:middle[1]+box_size, middle[0]-box_size:middle[0]+box_size] = 0
    while num_filled < 0.2*(len(img) * len(img[0])):
        num_filled, mask, _, _ = cv2.floodFill(hue.copy(), None, middle, 255, upDiff=diff, loDiff=diff, flags=cv2.FLOODFILL_FIXED_RANGE)
        diff += 1
    #_show(mask)

    hue[np.where(mask < 255)] = 0

    hue *= 255
    hue = hue.astype(np.uint8)
    ret, hue = cv2.threshold(hue, 1, 255, cv2.THRESH_BINARY)

    #_show(hue)

    im2, contours, hierarchy = cv2.findContours(hue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("test", r_g)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    polygon = contours[0]
    for i in range(1, len(contours)):
        polygon = np.concatenate((polygon, contours[i]))

    #polygon = np.where(hue == 255)
    #zipped = []
    #print(contours[0])
    #print(len(polygon[0]), len(polygon[1]))
    #for i in range(len(polygon[0])):
    #    zipped.append(((polygon[1][i], polygon[0][i]), ))

    #polygon = np.array(zipped)
    #print(polygon)
    # polygon = cv2.approxPolyDP(contours[0], 20, True)
    # polygon = cv2.approxPolyDP(polygon, 100, True)

    # return polygon
    return cv2.convexHull(polygon)

    _show(hue)
    return


    ret, hue = cv2.threshold(hue, 10, 255, cv2.THRESH_BINARY)

    iterations = 1
    size = 5
    erosionElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    hue = cv2.erode(hue, erosionElement, iterations=iterations)

    #_show(hue, "after erosion")

    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    hue = cv2.dilate(hue, dilationElement, iterations=iterations)

    #_show(hue, "after dilation")




    #_show(hue)
    return


    #_show(transformed)
    return
    #small = cv2.resize(hue, (0,0), fx=0.01, fy=0.01)
    #big = cv2.resize(small, (1920,1080))
    #big[np.where(big < 60)] = 0
    #big[np.where(big > 130)] = 0
    #hue[np.where(big == 0)] = 0

    _show(hue)

    blur = 7
    blurred = cv2.blur(hue, (blur,blur))
    blurred[np.where(blurred == 255)] = 254
    diff = 2
    num_filled = 0
    while num_filled < 0.5*(len(img) * len(img[0])):
        num_filled, mask, _, _ = cv2.floodFill(blurred.copy(), None, middle, 255, upDiff=diff, loDiff=diff)
        diff += 1
    hue[np.where(mask < 255)] = 0

    _show(hue)

    hue = threshold(hue)

    _show(hue)

    #hue = cv2.adaptiveThreshold(hue, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 115, 2)
    iterations = 3
    size = 5
    erosionElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    while middle_avg(hue, count_blacks=True) > 100:
        hue = cv2.erode(hue, erosionElement, iterations=1)

    _show(hue, "after erosion")

    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    while middle_avg(hue, count_blacks=True) < 200:
        hue = cv2.dilate(hue, dilationElement, iterations=1)

    _show(hue, "after dilation")
    #hue = threshold(hue.copy(), 30)


    fill_threshold(hue)
    _show(hue)

    im2, contours, hierarchy = cv2.findContours(hue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("test", r_g)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    polygon = contours[0]
    # polygon = cv2.approxPolyDP(contours[0], 20, True)
    # polygon = cv2.approxPolyDP(polygon, 100, True)

    # return polygon
    return cv2.convexHull(polygon)
    return

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b = lab[:,:,2]
    b = threshold(b, 10)
    _show(b)
    return

    blurred = cv2.blur(hue, (3,3))


    hue = threshold(hue)
    blurred = threshold(blurred, 10)
    #i = cv2.adaptiveThreshold(hsv[:,:,0], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 111, 8)

    _show(hue)
    # _show(blurred)

    middle = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    result = cv2.floodFill(hue, None, middle, 120)
    _, hue, _, _ = result

    _show(hue)

    hue[np.where(hue != 120)] = 0
    hue[np.where(hue != 0)] = 255

    _show(hue)

    im2, contours, hierarchy = cv2.findContours(hue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(hue, contours, 0, (255,255,0))
    # cv2.imshow("test", r_g)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    polygon = contours[0]
    # polygon = cv2.approxPolyDP(contours[0], 20, True)
    # polygon = cv2.approxPolyDP(polygon, 100, True)

    # return polygon
    return cv2.convexHull(polygon)

    return
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for i in (0,1,2):
        _show(lab[:,:,i])
    return
    for i in (0,1,2):
        _show(hsv[:,:,i])


    img[:,:,0] = 0
    _show(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return
    b, g, r = img[:,:,0].astype(int), img[:,:,1].astype(int), img[:,:,2].astype(int)


    b = (b * (255 / np.amax(b))).astype(np.uint8)
    _show(b)

    tot = b+g+r

    tot = (tot * (255 / np.amax(tot))).astype(np.uint8)
    _show(tot)

    return

    g_b = np.absolute(g - b)
    r_b = np.absolute(r - b)
    r_g = np.absolute(r - g)

    g_b = (g_b * (255 / np.amax(g_b))).astype(np.uint8)
    r_b = (r_b * (255 / np.amax(r_b))).astype(np.uint8)

    r_g = cv2.blur(r_g, (5,5))
    r_g = (r_g * (255 / np.amax(r_g))).astype(np.uint8)
    from matplotlib import pyplot as plt

    plt.hist(r_g.ravel(),256,[0,256]); plt.show()
    #hist = cv2.calcHist([r_g],[0],None,[256],[0,256])

    #g_b_avg = np.absolute(g_b - np.average(g_b))
    #r_b_avg = np.absolute(r_b - np.average(r_b))
    r_g_avg = np.absolute(r_g - np.average(r_g))

    #g_b_avg = (g_b_avg * (255 / np.amax(g_b_avg))).astype(np.uint8)
    #r_b_avg = (r_b_avg * (255 / np.amax(r_b_avg))).astype(np.uint8)
    r_g_avg = (r_g_avg * (255 / np.amax(r_g_avg))).astype(np.uint8)

    g_b_avg = g_b
    r_b_avg = r_b
    #r_g_avg = r_g

    g_b_avg = cv2.adaptiveThreshold(g_b_avg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, 12)
    r_b_avg = cv2.adaptiveThreshold(r_b_avg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, 12)
    r_g_avg = cv2.adaptiveThreshold(r_g_avg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 7)

    #_show(g_b_avg)
    #_show(r_b_avg)
    _show(r_g_avg)
    return
    _show(g_b, "abs(g_b)")
    _show(r_b, "abs(r_b)")
    _show(r_g, "abs(r-g)")
    return
    # r_g = cv2.blur(r_g, (5,5))
    r_g = cv2.GaussianBlur(r_g, (9,9), 0)
    _, r_g = cv2.threshold(r_g, 30, 255, cv2.THRESH_BINARY)

    _show(r_g, "after blur and threshold")

    # For some reason, some images have the field in a lighter color than the rest
    # so if the field has a higher average value than the rest of the picture, we invert the picture
    middle = (int(img.shape[1] / 2), int(img.shape[0] / 2))
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

    middle = (int(img.shape[0] / 2), int(img.shape[1] / 2))
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
