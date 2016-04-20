#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
import cv2
from os import walk


def cmask(center, radius, array):
    # print(center)
    # print(radius)
    a, b = center
    ny, nx, _ = array.shape
    # ny, nx = array.shape
    y, x = np.ogrid[-b:ny-b, -a:nx-a]
    return x*x + y*y <= radius*radius


def minimize_sum_of_squared_gradients(img):
    # cimg = img.copy()
    # cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    # circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=30, minRadius=7, maxRadius=25)
    # cv2.imshow("test", cimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # if circles is not None:
    #     for i in circles[0,:]:
    #         return (i[2], i[0], i[1])
    # return (0, 0, 0)
    # img = img.copy()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imshow("test", img[:,:,0])
    # cv2.waitKey(0)
    # cv2.imshow("test", img[:,:,1])
    # cv2.waitKey(0)
    # cv2.imshow("test", img[:,:,2])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    center_x, center_y = int(len(img[0]) / 2), int(len(img) / 2)

    x_first = img[:-1,:-1].astype(np.int64)
    x_last = img[:-1,1:].astype(np.int64)

    y_first = img[:-1,:-1].astype(np.int64)
    y_last = img[1:,:-1].astype(np.int64)

    x_gradient_squared = np.power(x_last - x_first, 2)
    y_gradient_squared = np.power(y_last - y_first, 2)

    min_radius, max_radius = 7, 15

    min_sum = float("INF")
    best_param = None
    for radius in range(min_radius, max_radius+1):
        for d_x in range(-10, 11):
            cur_x = center_x + d_x
            for d_y in range(-10, 11):
                cur_y = center_y + d_y

                this_sum = sum_of_squared_gradients(img, cmask((cur_x, cur_y), radius, img[:-1,:-1]),
                                                    x_gradient_squared, y_gradient_squared)
                if this_sum < min_sum:
                    min_sum = this_sum
                    best_param = (radius, cur_x, cur_y)

    print(min_sum, best_param)
    return min_sum, best_param[0], best_param[1], best_param[2]


    # Returns average of squared gradients
def sum_of_squared_gradients(img, mask=None, x_gradient_squared=None, y_gradient_squared=None):
    if len(img) == 0 or len(img[0]) == 0:
        return float("INF")

    if x_gradient_squared is None or y_gradient_squared is None:
        x_first = img[:-1,:-1].astype(np.int64)
        x_last = img[:-1,1:].astype(np.int64)

        y_first = img[:-1,:-1].astype(np.int64)
        y_last = img[1:,:-1].astype(np.int64)

        x_gradient_squared = np.power(x_last - x_first, 2)
        y_gradient_squared = np.power(y_last - y_first, 2)

    if mask is None:
        x_sum = np.sum(x_gradient_squared)
        y_sum = np.sum(y_gradient_squared)

        return (x_sum + y_sum) / (len(img[0])+len(img))

    else:
        # print()
        # print(img[mask])
        # print(img)

        sum_gradient = np.sum(x_gradient_squared[mask]) + np.sum(y_gradient_squared[mask])
        num_gradients = np.sum(mask)

        # We want to favor bigger matches, so we get a greedy search
        return sum_gradient / (((num_gradients/3.14)**0.5)**1.5)


if __name__ == "__main__":
    from image import Image
    import utilities
    image1 = Image("/home/anders/UNIK4690/project/images/microsoft_cam/24h/south/2016-04-12_18:43:04.png")
    image2 = Image("/home/anders/UNIK4690/project/images/microsoft_cam/24h/south/2016-04-12_19:10:03.png")

    bgr = image2.get_bgr()
    b = bgr[:,:,0]
    g = bgr[:,:,1]
    r = bgr[:,:,2]
    b_target_average = np.average(image1.get_bgr()[:,:,0])
    g_target_average = np.average(image1.get_bgr()[:,:,1])
    r_target_average = np.average(image1.get_bgr()[:,:,2])
    print(b_target_average, g_target_average, r_target_average)
    b_ratio = np.average(image1.get_bgr()[:,:,0]) / np.average(image2.get_bgr()[:,:,0])
    g_ratio = np.average(image1.get_bgr()[:,:,1]) / np.average(image2.get_bgr()[:,:,1])
    r_ratio = np.average(image1.get_bgr()[:,:,2]) / np.average(image2.get_bgr()[:,:,2])
    print(b_ratio, g_ratio, r_ratio)
    b *= b_ratio
    b[np.where(b > 1.0)] = 1.0
    g *= g_ratio
    g[np.where(g > 1.0)] = 1.0
    r *= r_ratio
    r[np.where(r > 1.0)] = 1.0
    image2.bgr = bgr

    utilities.show(image1.get_bgr(), "image1", text=image1.filename, draw_histograms=True, time_ms=1)
    utilities.show(image2.get_bgr(), "image2", text=image2.filename, draw_histograms=True)

    exit(0)

    filename = "IMG_20160330_155620.jpg"

    img = cv2.imread("images/1920x1080/" + filename)

    not_ball = img[690:719, 460:489]
    # plt.imshow(cv2.cvtColor(not_ball, cv2.COLOR_BGR2RGB), interpolation='bicubic')
    # plt.show()

    # ball = img[649:678, 436:465]
    ball = img[640:698, 430:485]
    # ball = img[688:698, 478:485]
    # plt.imshow(cv2.cvtColor(ball, cv2.COLOR_BGR2RGB), interpolation='bicubic')
    # plt.show()

    print(sum_of_squared_gradients(not_ball))
    print(sum_of_squared_gradients(ball))

    # cv2.imshow("test", img)
    # cv2.waitKey(0)

    radius, x, y = minimize_sum_of_squared_gradients(ball)
    cv2.circle(ball, (int(x), int(y)), radius, [255,0,0])
    cv2.imshow("test", ball)
    cv2.waitKey(0)
