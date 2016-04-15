import collections
import math
import numpy as np
import cv2
import utilities
import os
from utilities import read_metadata
from utilities import update_metadata

def extract_bb(img, bb):
    x,y,w,h = bb
    return img[y:y+h, x:x+w]

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def get_playground_mask(img_path):
    img = cv2.imread(img_path)
    meta = read_metadata(img_path)
    if "playground_poly" not in meta:
        playground_poly = utilities.select_polygon(img)
        meta = update_metadata(img_path, {"playground_poly": playground_poly})
    playground_mask = utilities.poly2mask(meta["playground_poly"], img)
    return playground_mask

def mean_diff(img, params):
    mean_playground = cv2.mean(img, params.fg_mask)[0]
    mean_background = cv2.mean(img, params.bg_mask)[0]
    diff = abs(mean_background - mean_playground)
    return diff

def deviation(img, params):
    mean, std_dev = cv2.meanStdDev(img, mask=params.fg_mask)
    return std_dev[0][0]

def abs_deviation(img, params):
    mean = cv2.mean(img, params.fg_mask)[0]
    diff = cv2.absdiff(img, mean)
    return cv2.sumElems(apply_mask(diff, params.fg_mask))[0] / params.fg_pixel_count

def histogram_compare(img, params):
    resolution = 100
    h1 = cv2.calcHist([img], [0], params.fg_mask, [resolution], [0, 1.0])
    h2 = cv2.calcHist([img], [0], params.bg_mask, [resolution], [0, 1.0])
    cv2.normalize(h1, h1, alpha=1, norm_type=1)
    cv2.normalize(h2, h2, alpha=1, norm_type=1)
    return cv2.compareHist(h1, h2, 0)
    # Similarity
    # cv2.normalize(h1, h1, alpha=1, norm_type=2)
    # cv2.normalize(h2, h2, alpha=1, norm_type=2)
    # return h1.reshape((resolution)).dot(h2.reshape((resolution)))

def discriminatory_power(img, params):
    """
    Taken from "Autonomous Robotic Vehicle Road Following, 1988"
    """
    mean1, std_dev1 = cv2.meanStdDev(extract_bb(img, params.fg_bb), mask=params.fg_mask_bb)
    mean2, std_dev2 = cv2.meanStdDev(img, mask=params.bg_mask)
    # Return format is a bit weird ...
    var1 = std_dev1[0][0]**2
    var2 = std_dev2[0][0]**2
    mean1 = mean1[0][0]
    mean2 = mean2[0][0]
    return (mean1 - mean2)**2 / (var1 + var2)

def create_img_set_fitness_function(img_paths, fg_mask):
    images = []
    for img_path in img_paths:
        bgr = cv2.imread(img_path)
        bgr = bgr.astype(np.float32)
        bgr /= 255

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        images.append((bgr, hsv, lab, ycrcb))

    N = len(images)

    bg_mask = cv2.bitwise_not(fg_mask)
    fg_pixel_count = cv2.countNonZero(fg_mask)
    fg_bb = cv2.boundingRect(fg_mask)
    fg_mask_bb = extract_bb(fg_mask, fg_bb)

    # Parameters that might be relevant to fitness metrics
    Params = collections.namedtuple("Params", ("fg_mask", "bg_mask", "fg_bb", "fg_mask_bb", "fg_pixel_count"))
    params = Params(fg_mask, bg_mask, fg_bb, fg_mask_bb, fg_pixel_count)

    timer = utilities.Timer()

    def fitness(transformer, parm):
        timer.reset()
        fitness_sum = 0
        for img in images:
            img = transformer(img, parm)
            # diff = mean_diff(img, mask, bg_mask)
            # inv_dev = 1-deviation(img, mask, fg_pixel_count)
            # inv_abs_dev = 1-abs_deviation(img, mask, fg_pixel_count)
            fitness_value = discriminatory_power(img, params)
            fitness_sum += fitness_value
            # fitness_sum += diff+inv_dev/2
            # img = img.copy()
            # cv2.putText(img, str(parm) + " " + str(diff)[:4] + ", " + str(inv_dev) + ", " + str(inv_abs_dev), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
            # cv2.putText(img, str(fitness_value), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
            # cv2.imshow("test", img)
            # utilities.wait_for_key('n')

        # print(timer)
        return (fitness_sum / N)

    return fitness

def create_fitness_function_v1(dir_path):
    img_names = ["2016-04-12_16:19:04.png", "2016-04-12_17:46:03.png", "2016-04-12_18:20:04.png", "2016-04-12_19:04:04.png",
                 "2016-04-12_19:09:03.png", "2016-04-12_20:34:05.png", "2016-04-13_05:54:09.png", "2016-04-13_08:17:04.png",
                 "2016-04-13_09:08:04.png", "2016-04-13_10:30:04.png"]
    img_paths = list(map(lambda name: os.path.join(dir_path, name), img_names))
    mask = get_playground_mask(img_paths[0])

    return create_img_set_fitness_function(img_paths, mask)

def convert_to_float(img):
    img = img.astype(np.float32)
    return img / np.max(img)

if __name__ == "__main__":
    fitness = create_fitness_function_v1("images/microsoft_cam/24h/south")

    def rgb_single_transformer(img, channel):
        return img[0][:,:,channel]

    def hsv_single_transformer(img, channel):
        return img[1][:,:,channel]

    from timeit import default_timer as timer

    start = timer()
    print(fitness(rgb_single_transformer, 0))
    print(fitness(rgb_single_transformer, 1))
    print(fitness(rgb_single_transformer, 2))
    print()
    print(fitness(hsv_single_transformer, 0))
    print(fitness(hsv_single_transformer, 1))
    print(fitness(hsv_single_transformer, 2))
    end = timer()
    print((end-start)/6*1000)
