#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
import cv2
from os import walk
import utilities


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
    from transformer import Transformer, mean_diff

    transformer = Transformer(filename="playground_transformer_state.json")
    try:
        import os
        filenames = []
        for cur in os.walk(os.path.join(utilities.get_project_directory(), "images/microsoft_cam/24h/south/")):
            filenames = cur[2]
            break

        filenames.sort()

        for file in filenames:
            try:
                import datetime
                date = datetime.datetime.strptime(file, "%Y-%m-%d_%H:%M:%S.png")
                if date < datetime.datetime(2016, 4, 13, 7, 5):
                # if date < datetime.datetime(2016, 4, 12, 19, 0):
                    continue
                image = Image(os.path.join(utilities.get_project_directory(), "images/microsoft_cam/24h/south/", file))

            except FileNotFoundError:
                continue

            size = 40
            best_transform = utilities.as_uint8(transformer.get_playground_transformation(image))
            box = utilities.get_box(best_transform.copy(), side=size)
            box_hist = utilities.get_histogram(box)

            tot = np.zeros(best_transform.shape[:2]).astype(np.float32)
            for row in range(0, 1080, size):
                for col in range(0, 1920, size):
                    box = best_transform[row:row+size, col:col+size]
                    cur_box_hist = utilities.get_histogram(box)
                    tot[row:row+size, col:col+size] = cv2.compareHist(box_hist, cur_box_hist, cv2.HISTCMP_CORREL)

            tot -= np.amin(tot)
            tot *= 1/np.amax(tot)
            utilities.show(tot, "Finished", time_ms=1)


            # utilities.show(best_transform, time_ms=10, text=img.filename)
    except Exception as e:
        import traceback
        print(e)
        traceback.print_tb(e.__traceback__)
