import numpy as np
import cv2
import utilities
import os
from utilities import read_metadata
from utilities import update_metadata

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

def mean_diff(img, mask1, mask2):
    mean_playground = cv2.mean(img, mask1)[0]
    mean_background = cv2.mean(img, mask2)[0]
    diff = abs(mean_background - mean_playground)
    return diff

def abs_deviation(img, mask, mask_pixel_count):
    mean = cv2.mean(img, mask)[0]
    diff = cv2.absdiff(img, mean)
    return cv2.sumElems(apply_mask(diff, mask))[0] / mask_pixel_count

def create_img_set_fitness_function(img_paths, mask):
    complement_mask = cv2.bitwise_not(mask)
    images = []
    for img_path in img_paths:
        images.append(cv2.imread(img_path))

    mask_pixel_count = cv2.countNonZero(mask)

    def fitness(transformer, parm):
        fitness_sum = 0
        for img_bgr in images:
            img = transformer(img_bgr, parm)
            diff = mean_diff(img, mask, complement_mask)
            inv_dev = 1-abs_deviation(img, mask, mask_pixel_count)
            fitness_sum += diff+inv_dev/2
            # img = img.copy()
            # cv2.putText(img, str(parm) + " " + str(diff)[:4] + ", " + str(inv_dev), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
            # print(diff)
            # cv2.imshow("test", img)
            # utilities.wait_for_key('n')

        return (fitness_sum / len(images))

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
    fitness = create_fitness_function_v1("images/series-1")

    def rgb_single_transformer(img, channel):
        return convert_to_float(img[:,:,channel])

    def hsv_single_transformer(img, channel):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return convert_to_float(img[:,:,channel])

    from timeit import default_timer as timer
    # fitness(rgb_single_transformer, 0)

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
