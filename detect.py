#!/usr/bin/python3
import numpy as np
import matplotlib as plt
import cv2
from os import walk
from test import minimize_sum_of_squared_gradients, sum_of_squared_gradients


# lower score is better
def calc_density_score(img, points):
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


def maximize_density(img):
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

    # Polygon:
    #            p3
    #   p0
    #
    #  p1      p2
    #

    cur_score = calc_density_score(img, [p0, p1, p2, p3])

    threshold = (width*height - np.count_nonzero(img)) / (0.8*width*height)

    while True:
        change = False
        new_p0 = (max(0, p0[0]-10), p0[1])
        new_p1 = (max(0, p1[0]-10), p1[1])
        if new_p0 != p0 or new_p1 != p1:
            delta_score = calc_density_score(img, [new_p0, new_p1, p1, p0])
            if delta_score < threshold:
                p0 = new_p0
                p1 = new_p1
                change = True

        new_p2 = (min(width-1, p2[0]+10), p2[1])
        new_p3 = (min(width-1, p3[0]+10), p3[1])
        if new_p2 != p2 or new_p3 != p3:
            delta_score = calc_density_score(img, [p2, p3, new_p3, new_p2])
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
            delta_score = calc_density_score(img, [new_p0, p0, p3, new_p3])
            if delta_score < threshold:
                p0 = new_p0
                p3 = new_p3
                change = True

        new_p1 = (p1[0], min(height-1, p1[1]+10))
        new_p2 = (p2[0], min(height-1, p2[1]+10))
        if new_p1 != p1 or new_p2 != p2:
            delta_score = calc_density_score(img, [p1, new_p1, new_p2, p2])
            if delta_score < threshold:
                p1 = new_p1
                p2 = new_p2
                change = True

        if not change:
            break


    cur_score = calc_density_score(img, [p0, p1, p2, p3])
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

                new_score = calc_density_score(img, [new_p0, p1, p2, p3])
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

                new_score = calc_density_score(img, [p0, new_p1, p2, p3])
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

                new_score = calc_density_score(img, [p0, p1, new_p2, p3])
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

                new_score = calc_density_score(img, [p0, p1, p2, new_p3])
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
        new_score = calc_density_score(img, [new_p, p1, p2, p3])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p0 = new_p
            cur_score = new_score
            change = True

        new_p = (p0[0], max(0, p0[1]-10))
        new_score = calc_density_score(img, [new_p, p1, p2, p3])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p0 = new_p
            cur_score = new_score
            change = True

        # p1
        new_p = (max(0, p1[0]-10), p1[1])
        new_score = calc_density_score(img, [p0, new_p, p2, p3])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p1 = new_p
            cur_score = new_score
            change = True

        new_p = (p1[0], min(height-1, p1[1]+10))
        new_score = calc_density_score(img, [p0, new_p, p2, p3])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p1 = new_p
            cur_score = new_score
            change = True

        # p2
        new_p = (min(width-1, p2[0]+10), p2[1])
        new_score = calc_density_score(img, [p0, p1, new_p, p3])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p2 = new_p
            cur_score = new_score
            change = True

        new_p = (p2[0], min(height-1, p2[1]+10))
        new_score = calc_density_score(img, [p0, p1, new_p, p3])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p2 = new_p
            cur_score = new_score
            change = True

        # p3
        new_p = (min(width-1, p3[0]+10), p3[1])
        new_score = calc_density_score(img, [p0, p1, p2, new_p])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p3 = new_p
            cur_score = new_score
            change = True

        new_p = (p3[0], min(0, p3[1]-10))
        new_score = calc_density_score(img, [p0, p1, p2, new_p])
        # print(cur_score, new_score)
        if new_score < cur_score:
            p3 = new_p
            cur_score = new_score
            change = True

        if not change:
            break

    return [p0, p1, p2, p3]


def detect_playing_field(img):
    r, g, b = img[:,:,0].astype(int), img[:,:,1].astype(int), img[:,:,2].astype(int)
    # r_g = b-r-g
    r_g = np.absolute(r - g)
    r_b = np.absolute(r - b)
    g_b = np.absolute(g - b)
    diff = np.maximum(np.maximum(r_g, r_b), g_b).astype(np.uint8)

    diff = cv2.blur(diff, (3,3))
    diff = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)

    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    erosionElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    diff = cv2.dilate(diff, dilationElement)
    # diff = cv2.erode(diff, erosionElement)
    diff = cv2.erode(diff, erosionElement)

    # diff = g_b.astype(np.uint8) * (255 / np.amax(g_b))
    # diff = cv2.blur(g_b, (3,3))
    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # erosionElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # diff = cv2.dilate(diff, dilationElement)
    # diff = cv2.erode(diff, erosionElement)
    # diff = cv2.erode(diff, erosionElement)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.imshow("r_g", r_g.astype(np.uint8) * (255 / np.amax(r_g)))
    # cv2.waitKey(0)
    # cv2.imshow("r_b", r_b.astype(np.uint8) * (255 / np.amax(r_b)))
    # cv2.waitKey(0)
    g_b = (g_b * (255 / np.amax(g_b))).astype(np.uint8)
    # kernel = np.array([0, 1, 0, 1, -4, 1, 0, 1, 0]).reshape((3,3))
    kernel = np.ones((3,3), np.float32)
    # g_b = cv2.filter2D(g_b, -1, kernel)

    # erosionElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))


    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    erosionElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    g_b = cv2.dilate(g_b, dilationElement, iterations=4)
    g_b = cv2.erode(g_b, erosionElement, iterations=4)

    g_b = cv2.adaptiveThreshold(g_b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 6)

    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    erosionElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    g_b = cv2.dilate(g_b, dilationElement, iterations=1)
    g_b = cv2.erode(g_b, erosionElement, iterations=5)

    # g_b = cv2.convertScaleAbs(g_b)
    # cv2.imshow("g_b", g_b)
    # cv2.waitKey(0)
    # im2, contours, hierarchy = cv2.findContours(g_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(g_b, contours, -1, (255,255,0), 3)
    # cv2.imshow("g_b", g_b)
    # cv2.waitKey(0)

    # kernel = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape((3,3)) / 9
    # g_b = cv2.filter2D(g_b, -1, kernel)

    # ret, g_b = cv2.threshold(g_b, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    points = maximize_density(g_b)

    for idx, point in enumerate(points):
        cv2.line(g_b, point, points[(idx+1)%len(points)], (0, 0, 255), 3)

    # cv2.imshow("diff", diff)
    # cv2.waitKey(0)
    cv2.imshow("g_b", g_b)
    cv2.waitKey(0)
    return
    im2, contours, hierarchy = cv2.findContours(g_b, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(g_b, contours, -1, (0,255,0), 3)
    cv2.imshow("g_b", g_b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

    avg = np.average(r_g)
    ret, r_g = cv2.threshold(r_g, 1.5*avg, 255, cv2.THRESH_BINARY)
    cv2.imshow("r-g", r_g)
    cv2.waitKey(0)
    # cv2.imshow("r", r)
    # cv2.waitKey(0)
    # cv2.imshow("g", g)
    # cv2.waitKey(0)
    # cv2.imshow("b", b)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    return
    img_diff = np.apply_along_axis(lambda x: np.max(x) - np.min(x) < 10, 2, img)
    img_diff = img_diff.astype(np.float)
    # img_mask = img_mask[..., np.newaxis]
    # print(img_mask)
    # print(img_mask.shape)

    img_mask = np.zeros([img_diff.shape[0], img_diff.shape[1], 3]).astype(np.float)
    img_mask[:,:,0] = img_diff
    img_mask[:,:,1] = img_diff
    img_mask[:,:,2] = img_diff

    cv2.imshow("test", img_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
    exit(0)
    copy = np.empty([x, y, 1])
    for i in range(len(copy)):
        for j in range(len(copy[i])):
            copy[i,j] = max(img[i,j][0], img[i,j][1], img[i,j][2]) - min(img[i,j][0], img[i,j][1], img[i,j][2])

    print(copy)
    exit(0)
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = HSV[:,:,0]
    S = HSV[:,:,1]
    V = HSV[:,:,2]

    S_blur = cv2.GaussianBlur(S, (5, 5), 0)
    # cv2.imshow("H", H)
    # cv2.waitKey(0)
    avg = np.average(S)
    ret, S = cv2.threshold(S, 1.5*avg, 255, cv2.THRESH_BINARY)
    cv2.imshow("S", S)
    cv2.waitKey(0)
    # cv2.imshow("S_blur", S_blur)
    # cv2.waitKey(0)
    # cv2.imshow("V", V)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()


def detection_method(img):
    S = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg = np.average(S)
    ret, S = cv2.threshold(S, 2*avg, 255, cv2.THRESH_BINARY)
    cv2.imshow("S", S)
    cv2.waitKey(0)


def show_keypoints(img):
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

    window_size = 5
    best_dark = [(None, float("INF"))] * 10
    best_light = [(None, float("INF"))] * 10
    for keypoint in kpDark:
        y,x = int(keypoint.pt[0]), int(keypoint.pt[1])
        this = sum_of_squared_gradients(img[y-window_size:y+window_size+1, x-window_size:x+window_size+1])
        if this < best_dark[-1][1]:
            best_dark[-1] = (keypoint, this)
            best_dark.sort(key=lambda x: x[1])

    for keypoint in kpLight:
        y,x = int(keypoint.pt[0]), int(keypoint.pt[1])
        this = sum_of_squared_gradients(img[y-window_size:y+window_size+1, x-window_size:x+window_size+1])
        if this < best_dark[-1][1]:
            best_light[-1] = (keypoint, this)
            best_light.sort(key=lambda x: x[1])

    print(best_dark)
    print(best_light)
    best_kp = [i[0] for i in best_dark if i[0] is not None]
    best_kp += [i[0] for i in best_light if i[0] is not None]

    keypointsImg = img.copy()
    cv2.drawKeypoints(img, kpDark, keypointsImg, color=[0, 0, 255])
    cv2.drawKeypoints(keypointsImg, kpLight, keypointsImg, color=[255, 0, 0])
    cv2.drawKeypoints(keypointsImg, best_kp, keypointsImg, color=[0, 255, 0])

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
            # img = cv2.imread("images/1920x1080/" + filename)
            img = cv2.imread("images/0.25/" + filename)
            # img = cv2.imread("images/800x600/" + filename)
            # img = cv2.imread("images/3840x2160/" + filename)
            # showKeypoints(img)
            detect_playing_field(img)
