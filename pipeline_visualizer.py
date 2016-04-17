
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

# parameter spec
# function: f(slider_ratio) -> value
# float range (0.0, 1.0)
# int range (1, 4)
# int range

def log(*args, **kwargs):
    print(*args, **kwargs)

class Operation:
    def __init__(self, name, op, params=None):
        self.name = name
        self.op = op
        if params is not None:
            self.params = {name: spec[0] for name, spec in params.items()}
            self.param_ranges = {name: spec[1] for name, spec in params.items()}
            self.param_specs = params
        else:
            self.params = None
    def has_params(self):
        return self.params is not None


def visualize_pipeline(start_img, operations, scale_denom=3, row_count=2):
    cv2.namedWindow("pipeline-viz")
    resolution = 100
    inputs = [None]*(len(operations)+1)
    inputs[0] = start_img

    h, w = start_img.shape[:2]
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

            run_pipeline(start_img, first_change=op)
            render_results()
        return callback

    def render_results():
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
        cv2.imshow("pipeline-viz", canvas)

    def run_pipeline(img, first_change=None):
        start_idx = 0 if first_change==None else operations.index(first_change)

        for i, op in enumerate(operations[start_idx:], start=start_idx):
            if op.params:
                img = op.op(inputs[i], **op.params)
            else:
                img = op.op(inputs[i])
        
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

    run_pipeline(start_img)
    render_results()
    utilities.wait_for_key('q')



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
        img = blur(img, 11)
        img = threshold(img, t, cv2.THRESH_TOZERO)
        return img

    for i in range(iterations):
        img = step(img)

    return img

iterative_blur_threshold_op = lambda: Operation("blur-threshold", iterative_blur_threshold,
                                                    {"t": (100, (0,256)), "iterations": (3, (0, 40))})


if __name__ == '__main__':

    img_paths = ["images/microsoft_cam/raw/1.jpg", "images/microsoft_cam/24h/south/latest.png"]
    images = list(map(cv2.imread, img_paths))

    select_image_op = lambda: Operation("select", lambda img, nr: pick_channel(to_hsv(images[nr]), 0),
                                        {"nr": (0, list(range(len(images))))})


    # IDEA: extract default parameter values using fun.__defaults__
    visualize_pipeline(pick_channel(to_hsv(images[0]), 0), 
        [
            select_image_op(),
            threshold_op(cv2.THRESH_TOZERO_INV),
            # make_repeated_op(blur_op(), 7),
            # threshold_op(cv2.THRESH_TOZERO),
            # make_repeated_op(blur_op(), 7),
            # threshold_op(cv2.THRESH_TOZERO),
            iterative_blur_threshold_op(),
            Operation("open", open, {"kernel_size": (3, [3,5,7,9],),
                                     "iterations" : (3, (1, 40))})
        ],
                       scale_denom=4, row_count=3
    )
