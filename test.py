#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
import cv2
from os import walk


def minimize_sum_of_squared_gradients(img):
    center_x, center_y = len(img[0]) / 2, len(img) / 2

    min_radius, max_radius = 15, 30

    min_sum = float("INF")
    best_param = None
    for radius in range(min_radius, max_radius+1):
        for d_x in range(-10, 11):
            cur_x = center_x + d_x
            for d_y in range(-10, 11):
                cur_y = center_y + d_y

                def dist_func(row, col):
                    import math
                    return math.sqrt((row-cur_y)**2 + (col-cur_x)**2) <= radius

                this_sum = sum_of_squared_gradients(img, dist_func)
                if this_sum < min_sum:
                    min_sum = this_sum
                    best_param = (radius, cur_x, cur_y)

    print(min_sum, best_param)
    return best_param


# Returns average of squared gradients
def sum_of_squared_gradients(img, dist_func=None):
    x_first = img[:,:-1].astype(np.int64)
    x_last = img[:,1:].astype(np.int64)

    y_first = img[:-1,:].astype(np.int64)
    y_last = img[1:,:].astype(np.int64)

    x_gradient_squared = np.power(x_last - x_first, 2)
    y_gradient_squared = np.power(y_last - y_first, 2)

    if dist_func is None:
        x_sum = np.sum(x_gradient_squared)
        y_sum = np.sum(y_gradient_squared)

        return (x_sum + y_sum) / (len(img[0])+len(img))

    else:
        sum_gradient = 0
        num_gradients = 0
        for row in range(len(img)):
            for col in range(len(img[0])):
                if dist_func(row, col):
                    if row < len(y_gradient_squared):
                        sum_gradient += y_gradient_squared[row, col]
                    if col < len(x_gradient_squared[0]):
                        sum_gradient += x_gradient_squared[row, col]
                    num_gradients += 1

        return np.sum(sum_gradient) / num_gradients


if __name__ == "__main__":
    filename = "IMG_20160330_155620.jpg"

    img = cv2.imread("images/1920x1080/" + filename)

    not_ball = img[690:719, 460:489]
    # plt.imshow(cv2.cvtColor(not_ball, cv2.COLOR_BGR2RGB), interpolation='bicubic')
    # plt.show()

    # ball = img[649:678, 436:465]
    ball = img[640:698, 430:485]
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
