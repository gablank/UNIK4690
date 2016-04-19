#!/usr/bin/python3
import cv2
import numpy as np
import math
import json
import os
import time
from math import ceil
import utilities
from functools import partial
from utilities import Timer

# parameter spec
# function: f(slider_ratio) -> value
# float range (0.0, 1.0)
# int range (1, 4)

class Operation:
    def __init__(self, name, op, params=None):
        self.name = name
        self.op = op
        if params is not None:
            self.params = {name: spec[0] for name, spec in params.items()}
            self.param_ranges = {name: spec[1] for name, spec in params.items()}
            self.param_specs = params # hack
        else:
            self.params = None
    def has_params(self):
        return self.params is not None


def visualize_pipeline(start_images, operations, scale_denom=3, row_count=2):
    cv2.namedWindow("pipeline-viz")

    def report_timing(*args):
        if True:
            print(*args)

    resolution = 100
    inputs = [None]*(len(operations)+1)

    if type(start_images) == list:
        def change_start_image(idx):
            inputs[0] = start_images[idx]
            run_pipeline()
            render_results()

        cv2.createTrackbar("source image", "pipeline-viz", 0, len(start_images)-1, change_start_image)
        inputs[0] = start_images[0]
    else:
        inputs[0] = start_images

    h, w = inputs[0].shape[:2]
    canvas_shape = (ceil(row_count * h/scale_denom), ceil((w//scale_denom) * ceil(len(inputs)/row_count)), 3)
    canvas = np.zeros(canvas_shape, dtype=np.uint8)

    def trackbarCallback(op, param_name):
        range = op.param_ranges[param_name]
        def callback(slider_value):
            ratio = slider_value/resolution
            value = None
            if type(range) == tuple:
                lo, hi = range
                if type(lo) == int:
                    value = lo+slider_value
                else:
                    value = ratio*(hi-lo) + lo
            elif type(range) == list:
                value = range[slider_value]
            op.params[param_name] = value

            import threading
            def change():
                run_pipeline(first_change=op)
                render_results()
            threading.Thread(target=change).start()
        return callback

    def render_results():
        timer = Timer()
        row = 0
        full_height, full_width = inputs[0].shape[:2]
        target_size = (full_width//scale_denom, full_height//scale_denom)
        w, h = target_size
        x,y = (0,0)
        row_size = ceil(len(inputs)/row_count)
        for i, img in enumerate(inputs):
            if i > 0 and (i % row_size) == 0:
                x = 0
                y += h
            img = cv2.resize(img, target_size)
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            canvas[y:y+h, x:x+w] = img[:,:]
            x += w

        report_timing("rendering took: %s" % (timer))
        cv2.imshow("pipeline-viz", canvas)

    def run_pipeline(first_change=None):
        start_idx = 0 if first_change==None else operations.index(first_change)
        timer = Timer()

        for i, op in enumerate(operations[start_idx:], start=start_idx):
            timer.reset()
            if op.params:
                img = op.op(inputs[i], **op.params)
            else:
                img = op.op(inputs[i])
            report_timing("%s took: %s with parameters: %s" % (op.name, timer, op.params))
        
            inputs[i+1] = img

    for i, op in enumerate(operations):
        if op.has_params():
            for name, range in op.param_ranges.items():
                count = None
                init_value = None
                if type(range) == list:
                    count = len(range)-1
                    init_value = range.index(op.params[name])
                elif type(range) == tuple:
                    lo, hi = range
                    if type(lo) == int:
                        count = hi-lo
                        init_value = op.params[name] - lo
                    else:
                        count = resolution
                        init_value = round((op.params[name]-lo)/(hi-lo) * count)

                cv2.createTrackbar(str(i)+" "+op.name + ":" +  name, "pipeline-viz",
                                   init_value, count, trackbarCallback(op, name))

    run_pipeline()
    render_results()

    def dump_parameters():
        print("{")
        for op in operations:
            print("\"%s\": %s," % (op.name, op.params))
        print("}")

    while True:
        key = utilities.wait_for_key()
        if key == 'd':
            dump_parameters()
        elif key == 'q':
            dump_parameters()
            exit(0)


def blur(img, size):
    if size==0:
        return img
    res = cv2.blur(img, (size, size))
    return res

blur_op = lambda: Operation("blur", blur, {"size": (3, [0,3,5,7,9,11])})

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

to_gray_op = lambda: Operation("to gray", to_gray)

def pick_channel(img, ch):
    return img[:,:,ch]

pick_channel_op = lambda: Operation("pick_channel", pick_channel, {"ch": (0, [0,1,2])})

def power(img, n):
    if n == 0:
        return img
    img = img / np.max(img)
    img = img**n
    img = np.round(img*255).astype(np.uint8)
    return img

power_op = lambda: Operation("power", power, {"n": (2, (0,16))})

def threshold_range(img, lo, hi):
    th_lo = cv2.threshold(img, lo, 255, cv2.THRESH_BINARY)[1]
    th_hi = cv2.threshold(img, hi, 255, cv2.THRESH_BINARY_INV)[1]
    return cv2.bitwise_and(th_lo, th_hi)

threshold_range_op = lambda: Operation("threshold_range", threshold_range,
                                       {"lo": (0, (0, 256)),
                                        "hi": (100, (0, 256))})

def threshold(img, t, threshold_type=cv2.THRESH_BINARY):
    return cv2.threshold(img, t, 255, threshold_type)[1]

threshold_op = lambda threshold_type: Operation("threshold",
                                                partial(threshold, threshold_type=threshold_type),
                                                {"t": (100, (0, 256))})

def close(img, iterations):
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    img = cv2.dilate(img, kernel, iterations=iterations)
    img = cv2.erode(img, kernel, iterations=iterations)
    return img

close_op = lambda: Operation("close", close, {"iterations": (1, (0, 30))})

def open(img, kernel_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    img = cv2.erode(img, kernel, iterations=iterations)
    img = cv2.dilate(img, kernel, iterations=iterations)
    return img

def make_repeated_op(op, n):
    def repeat(img, **kwargs):
        for i in range(n):
            img = op.op(img, **kwargs)
        return img
    return Operation(op.name, repeat, params=op.param_specs)

def iterative_blur_threshold(img, iterations, t):
    def step(img):
        img = blur(blur(img, 11), 7)
        img = threshold(img, t, cv2.THRESH_TOZERO)
        return img

    for i in range(iterations):
        img = step(img)

    return img

iterative_blur_threshold_op = lambda: Operation("blur-threshold", iterative_blur_threshold,
                                                {"t": (100, (0,256)), "iterations": (3, (0, 40))})

def dilate(img, size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    img = cv2.dilate(img, kernel, iterations=iterations)
    return img

dilate_op = lambda: Operation("dilate", dilate, {"size": (15, (0,30)),
                                                 "iterations": (2, (0, 10))})

def flood_fill_until(img, box_size, min_ratio):
    img = img.copy()
    middle = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    img[np.where(img == 255)] = 254
    img[middle[1]-box_size:middle[1]+box_size, middle[0]-box_size:middle[0]+box_size] = 0
    img = utilities.flood_fill_until(img, min_ratio)
    return img

flood_fill_until_op = lambda: Operation("flood fill until", flood_fill_until,
                                        {"box_size": (30, (0,100)),
                                         "min_ratio": (0.2, (0.0, 1.0))})

def set_params(pipeline, op_param_map):
    """
    Accepts the output dumped from the gui
    """
    for op in pipeline:
        op.params = op_param_map[op.name]
    return pipeline


if __name__ == '__main__':

    img_paths = ["raw/1.jpg", "raw/2.jpg", "raw/3.jpg", "24h/south/latest.png", "24h/south/2016-04-12_18:59:03.png", "24h/south/2016-04-12_19:21:04.png"]
    images = list(map(cv2.imread, ["images/microsoft_cam/"+img_path for img_path in img_paths]))
    images = [i for i in images if i is not None]

    iterative_blur_pipeline = [
        threshold_op(cv2.THRESH_TOZERO_INV),
        iterative_blur_threshold_op(),
    ]

    ranged_threshold_pipeline = [
        power_op(),
        threshold_range_op(),
        close_op(),
        Operation("open", open, {"kernel_size": (3, [3,5,7,9],),
                                 "iterations" : (3, (1, 40))})
    ]

    ranged_threshold_pipeline = set_params(ranged_threshold_pipeline,
                                           {
                                               "power": {'n': 0},
                                               "threshold_range": {'hi': 146, 'lo': 98},
                                               "close": {'iterations': 5},
                                               "open": {'kernel_size': 5, 'iterations': 21},
                                           })

    flood_fill_pipeline = [
        power_op(),
        dilate_op(),
        flood_fill_until_op(),
    ]

    # TODO: Operation representation isn't ideal. See ideas, but it's probably
    #       a bit to gather just by polishing current approach too
    # TODO: Add non-visualized operations
    # IDEA: use parameter annotations to encode parameter specs (range, etc). 
    #       https://www.python.org/dev/peps/pep-3107/#parameters
    #       def my_op_fn(img, param1: (0,10) = start_val, ...)
    #       But not sure if the annotations and default values are mutable.
    #       If not the set_params (useful to set dumped parameters) might be hard
    #       to implement elgantely?
    # IDEA: extract default parameter values using fun.__defaults__
    # IDEA: click to only show image under mouse (enlarged)
    visualize_pipeline([pick_channel(to_hsv(img), 0) for img in images], 
                       flood_fill_pipeline,
                       scale_denom=4, row_count=3
    )
