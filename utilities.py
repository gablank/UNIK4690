#!/usr/bin/python3
import cv2
import numpy as np
import math
import json
import os
import time
import random


def get_middle(img):
    """
    Get the middle of the image
    Res: P: (x,y)
    """
    mid_x = int(round(img.shape[1] / 2, 0))
    mid_y = int(round(img.shape[0] / 2, 0))
    return (mid_x, mid_y)

def draw_label(to_show, text):
    """
    Draw a label with defaults (position, etc.) that should work in most cases
    """
    x_pos, y_pos = 20, 30
    padding = 2
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 1
    text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)

    cv2.rectangle(to_show, (x_pos-padding, y_pos-text_size[1]-padding), (x_pos+text_size[0]+padding, y_pos+padding), (0, 0, 0), cv2.FILLED)
    cv2.putText(to_show, text, (x_pos, y_pos), font_face, font_scale, (255, 255, 255), thickness)

def show(img, win_name="test", fullscreen=False, time_ms=0, text=None, draw_histograms=False):
    """
    Show img in a window
    """
    if fullscreen:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    to_show = img.copy()
    to_show = as_uint8(to_show)

    if text is not None:
        draw_label(to_show, text)

    if draw_histograms:
        n_channels = 0 if len(to_show.shape) < 3 else to_show.shape[2]

        x_pos, y_pos = 10, 50
        y_padding = 20

        if n_channels == 0:
            hist = draw_histogram(to_show)
            to_show[y_pos:y_pos+hist.shape[0], x_pos:x_pos+hist.shape[1]] = hist

        else:
            for i in range(n_channels):
                channel = to_show[:,:,i]
                hist = draw_histogram(channel)

                for j in range(n_channels):
                    to_show[y_pos:y_pos+hist.shape[0], x_pos:x_pos+hist.shape[1], j] = hist

                y_pos += hist.shape[1] + y_padding

    cv2.imshow(win_name, to_show)

    key = cv2.waitKey(time_ms)
    if key % 256 == ord('q'):
        exit(0)

    #cv2.destroyWindow(win_name)
    return chr(key%256)


def get_box(img, center, side):
    """
    P: (x,y)
    Get a square with side lengths side centered on center
    """
    first_side = int(round(side / 2, 0))
    second_side = side - first_side
    return img[center[1]-first_side:center[1]+second_side, center[0]-first_side:center[0]+second_side]


def get_angle(p1, c, p2):
    """
    P: (x,y)
    Get the angle between the lines c -> p1 and c -> p2 in degrees
    Treats the lines as being directionless, so the result will always be 0 <= angle <= 180
    """
    line_1 = (p1[0]-c[0], p1[1]-c[1], 0)
    line_2 = (p2[0]-c[0], p2[1]-c[1], 0)
    v1_u = line_1 / np.linalg.norm(line_1)
    v2_u = line_2 / np.linalg.norm(line_2)
    angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle_radians * 180 / math.pi


def flood_fill_until(img, limit, center=None, color=255, max_diff=100):
    """
    Increase the loDiff and upDiff incrementally until limit of the pixels in the image has been filled
    """
    seed = center
    if seed is None:
        seed = get_middle(img)

    diff = 0
    num_filled = 0
    mask_shape = (img.shape[0]+2, img.shape[1]+2)
    while num_filled < limit*img.size:# and diff < max_diff:
        mask = np.zeros(mask_shape).astype(np.uint8)
        num_filled, _, _, _ = cv2.floodFill(img, mask, seed, color, upDiff=diff, loDiff=diff, flags=cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (color << 8))
        diff += 1

    # Remove the excess pixels around
    mask = mask[1:-1,1:-1]
    return mask


def draw_convex_hull(img, convex_hull):
    copy = img.copy()
    for idx, pt1 in enumerate(convex_hull):
        if idx % 2 == 0:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        pt1 = pt1[0]
        pt1 = (pt1[0], pt1[1])
        pt2 = convex_hull[(idx + 1) % len(convex_hull)][0]
        pt2 = (pt2[0], pt2[1])
        cv2.line(copy, pt1, pt2, color, 3)
    return copy

def poly2mask(poly, size_or_img):
    size = size_or_img if type(size_or_img) != np.ndarray else size_or_img.shape[0:2]
    mask = np.zeros(size, dtype=np.uint8)
    if type(poly) == list:
        poly = np.array(poly)
    cv2.fillPoly(mask, [poly], 255)
    return mask

def wait_for_key(char=None):
    while(True):
        key_code_raw = cv2.waitKey()
        # http://stackoverflow.com/a/17284668/1517969
        key_code = key_code_raw % 256

        if char is None:
            return chr(key_code)
        elif key_code == ord(char):
            return chr(key_code)

def select_polygon(orig_img):
    """
    Interactively select a polygon. Add points with LB and finish with key "s"
    """
    polygon = []
    color = (0,0,255)
    def mouse_callback(ev, x, y, flags, param):
        img = orig_img.copy()

        if ev == cv2.EVENT_LBUTTONDOWN:
            polygon.append((x,y))

        for p1,p2 in zip(polygon, polygon[1:]):
            cv2.line(img, p1, p2, color, 2)

        if len(polygon) > 0:
            cv2.circle(img, polygon[0], 4, color)
            cv2.line(img, polygon[-1], (x,y), color, 1)

        if len(polygon) > 1:
            cv2.line(img, (x,y), polygon[0], color, 1)

        cv2.imshow("polygon-select", img)

    cv2.namedWindow("polygon-select")
    cv2.setMouseCallback("polygon-select", mouse_callback)
    cv2.imshow("polygon-select", orig_img)
    wait_for_key('s')
    cv2.destroyWindow("polygon-select")
    return polygon

def select_circles(img):
    orig_img = img.copy()
    canvas = orig_img.copy()

    circles = []
    center = None
    color = (0,0,255)
    # Make drag image (in qt viewer) work without triggering clicks:
    mouse_moved_flag = False 

    message = ""

    def draw(canvas, center, r):
        cv2.circle(canvas, center, r, color, 1)

    def redraw(canvas):
        draw_label(canvas, message)
        for center, r in circles:
            draw(canvas, center, r)
        return canvas

    def mouse_callback(ev, x, y, flags, param):
        nonlocal canvas
        nonlocal center
        nonlocal mouse_moved_flag

        canvas = redraw(orig_img.copy())

        r = round(math.sqrt((center[0]-x)**2 + (center[1]-y)**2) if center else 0)

        if ev == cv2.EVENT_LBUTTONDOWN:
            mouse_moved_flag = False
        elif ev == cv2.EVENT_LBUTTONUP and not mouse_moved_flag:
            if not center:
                center = (x,y)
            else:
                circles.append((center, r))
                draw(canvas, center, r)
                center = None
        elif ev == cv2.EVENT_MOUSEMOVE:
            mouse_moved_flag = True
            if not center:
                pass
            else:
                draw(canvas, center, r)

        cv2.imshow("circles-select", canvas)
        
    cv2.namedWindow("circles-select")
    cv2.setMouseCallback("circles-select", mouse_callback)
    cv2.imshow("circles-select", orig_img)

    state_labels = ['Move (jkli)', 'Grow/Shrink (jl)']
    state_transforms = [
        lambda x,y,r,dx,dy: ((x+dx, y+dy), r),
        lambda x,y,r,dr,_:  ((x, y), r+dr),
    ]
    state = 0
    active_transform = state_transforms[state]
    message = state_labels[state]
    while True:
        key = wait_for_key()
        if key == 'u':
            circles.pop()
        elif key == 'n':
            state = (state + 1) % len(state_labels)
            active_transform = state_transforms[state]
            message = state_labels[state]
        elif key == 's':
            break

        if len(circles) > 0:
            cur = circles[-1]
            print(cur)
            args = (*cur[0], cur[1])
            new_circle = None
            if key == 'j':
                new_circle = active_transform(*args, -1,  0)
            elif key == 'l':
                new_circle = active_transform(*args,  1,  0)
            elif key == 'i':
                new_circle = active_transform(*args,  0, -1)
            elif key == 'k':
                new_circle = active_transform(*args,  0,  1)
            if new_circle:
                circles[-1] = new_circle

        cv2.imshow("circles-select", redraw(orig_img.copy()))

    cv2.destroyWindow("circles-select")
    return circles

def select_rects(img):
    """
    Interactively select a number of rectangles. Add points with LB and finish with key "s".
    Return a list of the rectangles. Each represented by the selected (two) points.
    """
    orig_img = img.copy()
    rects = []
    rect = []
    color = (0,0,255)
    # Make drag image (in qt viewer) work without triggering clicks:
    mouse_moved_flag = False 
    def mouse_callback(ev, x, y, flags, param):
        nonlocal orig_img
        nonlocal rect
        nonlocal mouse_moved_flag
        img = orig_img.copy()
        if ev == cv2.EVENT_LBUTTONDOWN:
            mouse_moved_flag = False
        elif ev == cv2.EVENT_LBUTTONUP and not mouse_moved_flag:
            rect.append((x,y))
            if len(rect) == 2:
                cv2.rectangle(img, rect[0], (x,y), color, 2)
                rects.append(rect)
                rect = []
                orig_img = img
        elif ev == cv2.EVENT_MOUSEMOVE:
            mouse_moved_flag = True
            if len(rect) > 0:
                cv2.rectangle(img, rect[0], (x,y), color, 2)
            elif len(rect) == 0:
                cv2.line(img, (x,0), (x, img.shape[0]), color, 2)
                cv2.line(img, (0,y), (img.shape[1], y), color, 2)

        cv2.imshow("rects-select", img)
        
    cv2.namedWindow("rects-select")
    cv2.setMouseCallback("rects-select", mouse_callback)
    cv2.imshow("rects-select", orig_img)
    wait_for_key('s')
    cv2.destroyWindow("rects-select")
    return rects

def two_point_rect_to_bb(p1, p2):
    """
    Converts a rectangle represented by two points to a bounding box: a (x,y,w,h) tuple
    1----+    2----o    +----2    +----1      +-w--*
    |    | or |    | or |    | or |    | ->   h    |
    +----2    o----1    1----*    2----*    (x,y)--*
    """
    x = min(p1[0], p2[0])
    y = min(p1[1], p2[1])
    w = abs(p1[0] - p2[0])
    h = abs(p1[1] - p2[1])
    return (x, y, w, h)



from matplotlib import pyplot as plt
def plot_histogram(img, channels=[0], mask=None, colors=["b", "g", "r"], max=None):
    """
    Adds the histogram to the active matplotlib plot. Use plt.show() after to show the plot.
    """
    if(type(colors) == str):
        colors=[colors]
    max = np.max(img)+0.00001 if max is None else max
    for idx, ch in enumerate(channels):
        hist = cv2.calcHist([img], [ch], mask, [256], [0, max])
        hist = hist/sum(hist) # normalize so each bucket represents percentage of total pixels
        plt.plot(hist, colors[idx])

def get_histogram(single_channel_img):
    single_channel_img = as_uint8(single_channel_img)
    return cv2.calcHist([single_channel_img], [0], None, [256], [0, 256])


def draw_histogram(single_channel_img, max_height=256, padding=2):
    """
    Get an image of the histogram (to be painted onto other images etc)
    """
    single_channel_img = as_uint8(single_channel_img)
    hist = cv2.calcHist([single_channel_img], [0], None, [256], [0, 256])

    hist_img = np.ones((max_height+2*padding, 256 + 2*padding))
    hist_img = as_uint8(hist_img)
    hist /= np.amax(hist)
    hist *= max_height

    for i in range(256):
        x = i
        y = int(hist[i][0])
        pt1 = (padding+x, padding + max_height-y)
        pt2 = (padding+x, padding + max_height-1)

        cv2.line(hist_img, pt1, pt2, (0, 0, 0))
    return hist_img

def get_metadata_path(img_path):
    img_name = os.path.basename(img_path)
    img_dir = img_path
    is_dir_path = img_name.find(".") < 0
    if not is_dir_path:
        img_base = "".join(img_name.split(".")[:-1])
        img_dir = os.path.dirname(img_path)
    series_metadata_path = os.path.join(img_dir, "metadata.json")
    if is_dir_path or os.path.exists(series_metadata_path):
        return series_metadata_path
    else:
        return os.path.join(img_dir, img_base+".json")

def read_metadata(img_path):
    metadata_path = get_metadata_path(img_path)
    if os.path.exists(metadata_path):
        with open(metadata_path) as fp:
            meta_dict = json.load(fp)
            return meta_dict
    else:
        return {}

def update_metadata(img_path, new_meta_data):
    meta_dict = read_metadata(img_path)
    meta_dict.update(new_meta_data)
    metadata_path = get_metadata_path(img_path)
    with open(metadata_path, "w") as fp:
        json.dump(meta_dict, fp) # overwrites on error too... 
    return meta_dict


def transform_image(img_spaces, vec):
    """
    Return a list of transformations that have previously worked well
    The factor for b is always 1
    :param img_spaces: tuple of image spaces: (bgr, hsv, lab, YCrCb)
    """
    bgr, hsv, lab, ycrcb = img_spaces

    transformed = np.zeros(bgr.shape[:2])

    idx = 0
    b, g, r = bgr[:,:,0], bgr[:,:,1], bgr[:,:,2]
    # transformed += vec[idx] * b
    # idx += 1
    transformed += vec[idx] * g
    idx += 1
    transformed += vec[idx] * r
    idx += 1

    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    # transformed += vec[idx] * h
    # idx += 1
    transformed += vec[idx] * s
    idx += 1
    transformed += vec[idx] * v
    idx += 1

    l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    transformed += vec[idx] * l
    idx += 1
    transformed += vec[idx] * a
    idx += 1
    transformed += vec[idx] * b
    idx += 1

    y, cr, cb = ycrcb[:,:,0], ycrcb[:,:,1], ycrcb[:,:,2]
    transformed += vec[idx] * y
    idx += 1
    transformed += vec[idx] * cr
    idx += 1
    transformed += vec[idx] * cb
    idx += 1

    # Normalization
    res = transformed - np.amin(transformed)
    res /= np.amax(res)

    return as_float32(res)


def as_uint8(img):
    if img.dtype == np.uint8:
        return img

    if img.dtype not in (np.float32, np.float64):
        raise RuntimeError("Unknown dtype: {}".format(img.dtype))

    img /= np.amax(img)
    img *= 255
    img = np.around(img)
    return img.astype(np.uint8)


def as_float32(img):
    if img.dtype in (np.float32, np.float64):
        return img.astype(np.float32) / np.amax(img)

    if img.dtype != np.uint8:
        raise RuntimeError("Unknown dtype: {}".format(img.dtype))

    img = img.astype(np.float32)
    return img / 255


def astype(img, dtype):
    if dtype == np.float32:
        return as_float32(img)

    if dtype == np.uint8:
        return as_uint8(img)

    raise RuntimeError("Unknown dtype: {}".format(dtype))


class Timer:
    def __init__(self):
        self.start = time.time()

    def reset(self):
        self.start = time.time()

    def __str__(self):
        return str(round(time.time() - self.start, 3)) + "s"


def get_project_directory():
    """
    Get an absolute path to the projects root directory
    """
    cur_dir = os.path.dirname(__file__)
    while cur_dir != "/" and not os.path.exists(os.path.join(cur_dir, ".git")):
        cur_dir = os.path.dirname(cur_dir)
    if not os.path.exists(os.path.join(cur_dir, ".git")):
        raise FileNotFoundError("Unable to locate project directory!")
    return cur_dir


def locate_file(path):
    """
    Attempt to locate the file path in the project.
    The path may be partial, as in:
    path = microsoft_cam/24h/south/2016-04-12_19:21:04.png
    will result in
    /home/anders/UNIK4690/project/images/microsoft_cam/24h/south/2016-04-12_19:21:04.png
    being returned (on my computer).
    path = 2016-04-12_19:21:04.png would yield the same result (as no file called 2016-04-12_19:21:04.png exists
    in any directory below /home/anders/UNIK4690/project/ except for in /home/anders/UNIK4690/project/images/microsoft_cam/24h/south/)

    If there are duplicates, the first encountered by os.walk is returned.
    """
    project_dir = get_project_directory()
    for root, dirs, files in os.walk(project_dir):
        # Remove directories we know can't contain the image
        dir_copy = list(dirs)
        for dir in dir_copy:
            if dir[0] == "." or dir == "__pycache__":
                dirs.remove(dir)

        # print("Looking for {} in {}".format(path, root))
        if os.path.exists(os.path.join(root, path)):
            path = os.path.join(root, path)
            break
    return path


if __name__ == "__main__":
    img_paths = ["raw/1.jpg", "raw/2.jpg", "raw/3.jpg", "24h/south/latest.png", "24h/south/2016-04-12_18:59:03.png", "24h/south/2016-04-12_19:21:04.png", "24h/south/2016-04-13_09:03:03.png", "24h/south/2016-04-13_12:45:04.png"]
    images = list(map(cv2.imread, ["images/microsoft_cam/"+img_path for img_path in img_paths]))

    images = [i for i in images if i is not None]

    # 90
    print(get_angle((0,100), (0,0), (100,0)))

    # 90
    print(get_angle((0,-100), (0,0), (-100,0)))

    # 90
    print(get_angle((0,100), (100,100), (100,0)))

    # ~0
    print(get_angle((0,100), (-100,-100), (0,100)))

    # ~180
    print(get_angle((-100,100), (0,0), (100,-100)))
