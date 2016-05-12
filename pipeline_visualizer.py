#!/usr/bin/python3
import cv2
import numpy as np
import math
import json
import os
from glob import glob
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
        if False:
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

    def mouseCallback(ev, x, y, flags, param):
        # Draw a marker in each image corresponding to the mouse pointer
        # Figure out which image the mouse is in
        # Calculate the relative coordinates
        
        pass

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

            run_pipeline(first_change=op)
            render_results()
        return callback

    def render_results():
        timer = Timer()
        row = 0
        full_height, full_width = inputs[0].shape[:2]
        target_size = (int(full_width//scale_denom), int(full_height//scale_denom))
        w, h = target_size
        x,y = (0,0)
        row_size = ceil(len(inputs)/row_count)
        canvas = np.zeros(canvas_shape, dtype=np.uint8)

        for i, img in enumerate(inputs):
            if i > 0 and (i % row_size) == 0:
                x = 0
                y += h
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
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
            names_sorted = sorted(op.param_ranges.keys())
            for name in names_sorted:
                parm_range = op.param_ranges[name]
                count = None
                init_value = None
                if type(parm_range) == list:
                    count = len(parm_range)-1
                    init_value = parm_range.index(op.params[name])
                elif type(parm_range) == tuple:
                    lo, hi = parm_range
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

def to_ycrcb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

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
def threshold_window(img, t, width):
    return threshold_range(img, t, t+width)

threshold_window_op = lambda: Operation("threshold_window", threshold_window,
                                        {"t": (100, (0,256)),
                                         "width": (10, (1, 256))})

def threshold(img, t, threshold_type=cv2.THRESH_BINARY):
    if True:
        t = round(np.max(img)*t/255)
    return cv2.threshold(img, t, 255, threshold_type)[1]

threshold_op = lambda threshold_type=cv2.THRESH_BINARY: Operation("threshold",
                                                partial(threshold, threshold_type=threshold_type),
                                                {"t": (100, (0, 256))})



def morph_close(img, iterations):
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    img = cv2.dilate(img, kernel, iterations=iterations)
    img = cv2.erode(img, kernel, iterations=iterations)
    return img

close_op = lambda: Operation("close", morph_close, {"iterations": (1, (0, 30))})

def morph_open(img, kernel_size, iterations):
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
    kernel = np.array([[2, 2, 2, 2, 2],
                       [2, 1, 1, 1, 2],
                       [2, 1, 1, 1, 2],
                       [2, 1, 1, 1, 2],
                       [2, 2, 2, 2, 2]])
    kernel = kernel/np.sum(kernel)
    def step(img):
        # img = cv2.medianBlur(img, 7)
        # img = cv2.filter2D(img, -1, kernel)
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

def set_pipeline_parameters(pipeline, op_param_map):
    """
    Accepts the output dumped from the gui
    """
    for op in pipeline:
        if op.name in op_param_map:
            op.params = op_param_map[op.name]
    return pipeline



iterative_blur_pipeline = [
    threshold_op(cv2.THRESH_TOZERO_INV),
    iterative_blur_threshold_op(),
]

ranged_threshold_pipeline = [
    power_op(),
    threshold_range_op(),
    close_op(),
    Operation("open", morph_open, {"kernel_size": (3, [3,5,7,9],),
                                "iterations" : (3, (1, 40))})
]

ranged_threshold_pipeline = set_pipeline_parameters(ranged_threshold_pipeline,
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


def gradient_length(img, size, order):
    img = img.astype(np.float32)
    kx1, kx2 = cv2.getDerivKernels(1, 0, size)
    ky1, ky2 = cv2.getDerivKernels(0, 1, size)
    Ix = cv2.sepFilter2D(img, -1, kx1, kx2)
    Iy = cv2.sepFilter2D(img, -1, ky1, ky2)

    return utilities.as_uint8(np.sqrt(Ix**2+Iy**2)) # or just abs(Ix+Iy)?

def gradient(img, size, order, which):
    dx = [order, 0, order]
    dy = [0, order, order]
    kx, ky = cv2.getDerivKernels(dx[which], dy[which], size)
    # img = img.astype(np.float32)
    img = cv2.sepFilter2D(img, cv2.CV_32F, kx, ky)
    return utilities.as_uint8(np.abs(img))
    return img

def distance(img, distance_type):
    return cv2.distanceTransform(img, distance_type, cv2.DIST_MASK_PRECISE)

def phase(img, size, order, norm_t):
    img = img.astype(np.float32)
    kx1, kx2 = cv2.getDerivKernels(1, 0, size)
    ky1, ky2 = cv2.getDerivKernels(0, 1, size)
    Ix = cv2.sepFilter2D(img, -1, kx1, kx2)
    Iy = cv2.sepFilter2D(img, -1, ky1, ky2)

    phase = cv2.phase(Ix, Iy)+1
    norm = np.sqrt(Ix**2+Iy**2)
    norm = norm / np.max(norm)

    hsv = np.ones((img.shape[0], img.shape[1], 3), dtype=np.float32)

    phase -= np.min(phase)

    hsv[:,:,0] = phase/np.max(phase)
    hsv[:,:,1] = 0.5
    hsv[:,:,2] = norm

    hsv *= 255
    hsv = np.round(hsv).astype(np.uint8)

    print(np.min(phase), np.max(phase))
    norm_max = np.max(norm)
    print(norm_max)
    # phase[np.where(norm < norm_t*norm_max)] = 0

    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return utilities.as_uint8(res) # or just abs(Ix+Iy)?

kernel_parm = (3, list(range(3, 50, 2)))

def cross_response(img, size, width):
    kernel = np.ones((size, size))
    m = size // 2
    w = round((m-1)*width)
    print(w, m, size, width)
    kernel[0:size, m-w:m+w+1] = -1
    kernel[m-w:m+w+1, 0:size] = -1
    kernel = kernel / np.sum(kernel[np.where(kernel > 0)])
    img = cv2.filter2D(img, -1, kernel)
    return img

cross_response_op = lambda: Operation("cross response", cross_response, {"size": kernel_parm, "width": (0.3, (0.0, 1.0))})

phase_pipeline = [
    blur_op(),
    Operation("phase", phase, {"size": (3, [3, 5, 7, 9, 11, 13, 15, 17, 31]),
                                "order": (1, [1,2,3,4]),
                                "norm_t": (0, (0.0, 1.0)),}),
    threshold_window_op(),
    # Operation("hist", utilities.draw_histogram),
    # threshold_op(cv2.THRESH_BINARY_INV),
]

gradient_pipeline = [
    blur_op(),
    Operation("grad", gradient, {"size": (3, [1, 3, 5, 7]), "order": (1, [1,2,3,4]), "which": (0, [0, 1, 2])}),
    # threshold_op(cv2.THRESH_BINARY_INV),
    cross_response_op(),
    # Operation("distance", distance, {"distance_type": (cv2.DIST_L1, [cv2.DIST_L1, cv2.DIST_L2, cv2.DIST_L12, cv2.DIST_C])}),
    # threshold_op(),

    # Operation("gradient-length", gradient_length, {"size": (3, [3, 5, 7, 9, 11]), "order": (1, [1,2,3,4])}),
]


def perspective_warp(img, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    M = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9]).reshape(3,3)

    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

warp_parms = { "x"+str(i): (0, (-30.0, 30.0)) for i in range(1,10) }


warp_pipeline = [
    Operation("warp", perspective_warp, warp_parms) 
]


def hough(img, min_radius, max_radius, p1, p2):
    if max_radius == 0: # hack to disable since it's sometimes very expensive
        return img
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=p1, param2=p2, minRadius=min_radius, maxRadius=max_radius)

    grayscale = img.copy()
    grayscale = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(grayscale,(i[0],i[1]),i[2],(0,255,0),3)
            # draw the center of the circle
            cv2.circle(grayscale,(i[0],i[1]),2,(0,0,255),3)

    return grayscale

hough_op = lambda: Operation("hough", hough, {"min_radius": (5, (1, 50)), "max_radius": (0, (0, 50)),
                                              "p1": (10, (1, 500)), "p2": (10, (1, 500))})

playing_balls_pipeline = set_pipeline_parameters([to_gray_op(), blur_op(), hough_op()],
                                   {
                                       "blur": {'size': 11},
                                       "hough": {'p1': 57, 'p2': 8, 'min_radius': 5, 'max_radius': 14},
                                   })

from utilities import pretty_print_keypoint

def make_keypoint_viz(fn):
    def wrapper(img, **kwargs):
        kps = fn(img, **kwargs)
        out = cv2.drawKeypoints(img, kps, None, color=[0, 0, 255],
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return out
    return wrapper

from playground_detection.red_balls import blob_detector, surf_detector

blob_op = Operation("blob", make_keypoint_viz(blob_detector), {
            "minArea": (10, (0,500)),
            "maxArea": (500, (0,2000)),
            "minDistBetweenBlobs": (100, (0,1000)),
            "blobColor" : (255, (0,255))
        })

from playground_detection.red_balls import red_ball_transform, normalize_image

def transform(img, a=0.5):
    img = img.astype(np.float32)
    diff = img[:,:,1] - a*img[:,:,2]
    diff = np.clip(diff, 0, 255)
    return utilities.as_uint8(diff)


# assumes ycrcb image as input
marker_balls_pipeline2 = set_pipeline_parameters(
    [
        # Operation("trans2", transform, {"a": (0.5, (0.0, 2.0))}),
        Operation("trans", red_ball_transform, {"exponent": (1, (1.0, 10.0))}),
        # blob_op,
        Operation("surf", make_keypoint_viz(surf_detector), {"hess_thresh": (300, (100, 10000))})
    ],
    {
        "trans": {'exponent': 5.0},
        "blob": {'blobColor': 255, 'minArea': 80, 'minDistBetweenBlobs': 100, 'maxArea': 1000},
    }
)

marker_balls_pipeline3 = set_pipeline_parameters(
    [
        pick_channel_op(),
        # Operation("trans", red_ball_transform, {"exponent": (5, (1.0, 10.0))}),
        Operation("normalize", normalize_image),
        # threshold_op(cv2.THRESH_BINARY),
        Operation("surf", make_keypoint_viz(surf_detector), {"hess_thresh": (1000, (1000, 20000))})
    ],
    {
        "pick_channel": {'ch': 1},
        "threshold": {'t': 207},
        "surf": {'hess_thresh': 10000},
    }
)

marker_balls_pipeline1 = set_pipeline_parameters(
    [pick_channel_op(),
     blur_op(),
     threshold_op(),
     hough_op(),
    ],
    {
        "pick_channel": {'ch': 1},
        "blur": {'size': 11},
        "threshold": {'t': 86},
        "hough": {'p2': 10, 'max_radius': 24, 'p1': 10, 'min_radius': 4},
    }
)



def extract_rectangle(img, r):
    x,y,w,h = r
    return img[y:y+h, x:x+w]

def extract_roi(img, region, mask_color=None):
    region = np.array(region)
    bb = cv2.boundingRect(region)
    x, y, w, h = bb
    roi = extract_rectangle(img, bb)
    if mask_color is not None:
        mask = utilities.poly2mask(region-(x,y), roi)
        roi[np.where(mask == 0)] = mask_color
    return roi 
                                   

def read_imgs(base_dir, paths):
    return list(map(cv2.imread, [base_dir+img_path.strip() for img_path in paths]))

def read_imgs_glob(pattern):
    return read_imgs("", glob(pattern))

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        img_paths = sys.argv[1:]
        imgs = read_imgs("", img_paths)
    else:
        with open("images/dual-lifecam,raspberry/raspberry-broken-red-balls-playground.result") as path_file:
            imgs = read_imgs("", path_file.readlines()[:6])

        imgs += read_imgs("", ["images/raspberry/may-2/raspberry-2016-05-02_12:01:35.png"])
        imgs += read_imgs_glob("images/dual-lifecam,raspberry/lifecam/lifecam-*png")

    visualize_pipeline(
        # raspberry_images,
        list(map(to_ycrcb, imgs)),
        # imgs,

        # warp_pipeline,
        # flood_fill_pipeline,
        # iterative_blur_pipeline,
        # gradient_pipeline,
        # phase_pipeline,
        # marker_balls_pipeline1,
        # marker_balls_pipeline2,
        marker_balls_pipeline3,
        # playing_balls_pipeline,
        # scale_denom=0.25, row_count=2
        scale_denom=3, row_count=2
    )

    # TODO: Operation representation isn't ideal. See ideas, but it's probably
    #       a bit to gather just by polishing current approach too
    # TODO: Add non-visualized operations
    # IDEA: use parameter annotations to encode parameter specs (range, etc). 
    #       https://www.python.org/dev/peps/pep-3107/#parameters
    #       def my_op_fn(img, param1: (0,10) = start_val, ...)
    #       But not sure if the annotations and default values are mutable.
    #       If not the set_pipeline_parameters (useful to set dumped parameters) might be hard
    #       to implement elgantely?
    # IDEA: extract default parameter values using fun.__defaults__
    # IDEA: click to only show image under mouse (enlarged)
    # hue_images = [pick_channel(to_hsv(img), 0) for img in images]
    # gray_images = [to_gray(img) for img in A_imgs]
    # #pg_images = [extract_roi(img, metadata["playground_poly"], (0)) for img in B_imgs]
    # pg_images = list(map(to_ycrcb, B_imgs))

# CRAP

    # balls = [utilities.extract_circle(img, ball_circle, 30) for ball_circle in metadata["ball_circles"]]
    # balls = list(map(to_gray, balls))

    # to_gray(cv2.imread("twisted_checkerboard.png"))

    # images = list(map(cv2.imread, ["images/microsoft_cam/"+img_path for img_path in img_paths]))
    # images = [i for i in images if i is not None]

    # base_dir = "images/microsoft_cam/24h/south/"

    # A = [ "2016-04-12_18:59:03.png", "2016-04-12_19:21:04.png",
    #              "2016-04-12_18:56:04.png" ]
    # A_imgs = read_imgs(base_dir, A)

    # B_imgs = [cv2.imread(path) for path in
    #           list(glob("images/red_balls/series-1/*.png"))[0::7]]


    # metadata = utilities.read_metadata("images/red_balls/series-1")

    # img = cv2.imread(base_dir+A[0])

    # raspberry_images = read_imgs_glob("images/dual-lifecam,raspberry/raspberry/*.png")

    # img_paths = ["raw/1.jpg", "raw/2.jpg", "raw/3.jpg", "24h/south/latest.png", "24h/south/2016-04-12_18:59:03.png", "24h/south/2016-04-12_19:21:04.png",
    #              "24h/south/2016-04-12_18:56:04.png", 
    # ]
