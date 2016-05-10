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

    while True:
        key = cv2.waitKey(time_ms)
        if key % 256 == ord('q'):
            exit(0)

        if time_ms > 0:
            break

        if key != 1114091 and key != 1114089: # <win>, <alt> (used to move/resize windows)
            break

    #cv2.destroyWindow(win_name)
    return chr(key%256)


def get_box(img, center=None, side=100):
    """
    P: (x,y)
    Get a square with side lengths side centered on center
    """
    if center is None:
        center = get_middle(img)
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
    """
    Interactively select a number of circles. Add center with LB and approx. radius with
    LB again. Then adjust position and radius with the keyboard.
    Return a list of the circles. Each represented by a tuple: ((cx, cy), r)
    """
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

    # Over-engineering ftw :) (This might make sense for arbitrary rectange selections though)
    state_labels = ['Move (jkli)', 'Grow/Shrink (jl)']
    state_transforms = [
        lambda x,y,r,dx,dy: ((x+dx, y+dy), r),
        lambda x,y,r,dr,_:  ((x, y), r+dr),
    ]
    state = 0
    active_transform = state_transforms[state]
    # message = state_labels[state]
    message = 'Move (jkli), Grow/Shrink (ed), Done (s), Quit (q)'
    while True:
        key = wait_for_key()
        if key == 'u':
            circles.pop()
        elif key == 'n' and False:
            state = (state + 1) % len(state_labels)
            active_transform = state_transforms[state]
            message = state_labels[state]
        elif key == 's':
            break
        elif key == 'q':
            exit(0)

        if len(circles) > 0:
            cur = circles[-1]
            (x,y), r = cur
            new_circle = None
            if key == 'j':
                new_circle = active_transform(x, y ,r , -1,  0)
            elif key == 'l':
                new_circle = active_transform(x, y ,r ,  1,  0)
            elif key == 'i':
                new_circle = active_transform(x, y ,r ,  0, -1)
            elif key == 'k':
                new_circle = active_transform(x, y ,r ,  0,  1)
            elif key == 'e':
                new_circle = (cur[0],cur[1]+1) # radius
            elif key == 'd':
                new_circle = (cur[0],cur[1]-1) # radius

            if new_circle:
                circles[-1] = new_circle

        cv2.imshow("circles-select", redraw(orig_img.copy()))

    cv2.destroyWindow("circles-select")
    return circles


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


def draw_histogram(single_channel_img, max_height=256, padding=2,
                   ignored_values=[], fg=0, bg=255, mask=None):
    """
    Get an image of the histogram (to be painted onto other images etc)
    """
    single_channel_img = as_uint8(single_channel_img)
    hist = cv2.calcHist([single_channel_img], [0], mask, [256], [0, 256])

    value_type = np.uint8
    if type(fg) != int or type(bg) != int:
        value_type = np.float32

    hist[ignored_values] = 0
    hist_img = np.ones((max_height+2*padding, 256 + 2*padding), dtype=value_type)*bg
    hist /= np.amax(hist)
    hist *= max_height

    for i in range(256):
        x = i
        y = int(hist[i][0])
        pt1 = (padding+x, padding + max_height-y)
        pt2 = (padding+x, padding + max_height-1)

        cv2.line(hist_img, pt1, pt2, (fg))
    return hist_img


def distance_point_to_line(line_pt1, line_pt2, pt):
    """
    Get the distance from pt to the line passing through line_pt1 and line_pt2
    """
    line_length = distance(line_pt1, line_pt2)
    if line_length == 0:
        return distance(line_pt1, pt)

    return abs((line_pt2[1]-line_pt1[1])*pt[0] - (line_pt2[0] - line_pt1[0])*pt[1] + line_pt2[0]*line_pt1[1] - line_pt2[1]*line_pt1[0])\
           / line_length


def distance_point_to_bounded_line(line_pt1, line_pt2, pt):
    """
    Get the distance from pt to the line passing through line_pt1 and line_pt2.
    The line is bounded at line_pt1 and line_pt2, so if the closest point is "outside" the line, the distance to the
    nearest point is returned.
    Example:
             line_pt2
                 o
                /
               /        pt
              /        o
             /
            /
           o
        line_pt1
    In this case the same number as returned from distance_point_to_line is returned

                          pt
                          o
             line_pt2
                 o
                /
               /
              /
             /
            /
           o
        line_pt1
    In this case the euclidean distance from pt to line_pt2 is returned
    """
    dist_infinite_line = distance_point_to_line(line_pt1, line_pt2, pt)
    euclidean_dist_to_line_pt1 = distance(line_pt1, pt)
    euclidean_dist_to_line_pt2 = distance(line_pt2, pt)
    distance_along_line_to_line_pt1 = math.sqrt(euclidean_dist_to_line_pt1**2 - dist_infinite_line**2)
    distance_along_line_to_line_pt2 = math.sqrt(euclidean_dist_to_line_pt2**2 - dist_infinite_line**2)
    line_length = distance(line_pt1, line_pt2)

    if distance_along_line_to_line_pt1 > line_length or distance_along_line_to_line_pt2 > line_length:
        return min(distance_along_line_to_line_pt1, distance_along_line_to_line_pt2)
    return dist_infinite_line


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


def transform_image(image, vec):
    """
    :param image: Image object
    :param vec: dict where keys = 1D color space (with preceding image space name: bgr_b, bgr_g etc.), and the value
    is the coefficient to multiply that color space by
    """
    transformed = None

    for image_space_name, image_space_data in image.get_color_space_dict().items():
        if transformed is None:
            transformed = np.zeros(image_space_data.shape[:2])

        if image_space_name in vec:
            transformed += vec[image_space_name] * image_space_data

    # Normalization
    res = transformed - np.amin(transformed)
    res /= np.amax(res)

    return as_float32(res)


def as_uint8(img):
    img = img.copy()
    if img.dtype == np.uint8:
        return img

    if img.dtype not in (np.float32, np.float64):
        raise RuntimeError("Unknown dtype: {}".format(img.dtype))

    img -= np.amin(img)
    img *= 1/np.amax(img)
    img *= 255
    img = np.around(img)
    return img.astype(np.uint8)


def as_float32(img):
    img = img.copy()
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


def show_all(image, time_ms=0):
    bgr = image.get_bgr()
    hsv = image.get_hsv()
    lab = image.get_lab()
    ycrcb = image.get_ycrcb()
    spaces = []
    for i in range(3):
        spaces.append(bgr[:,:,i])
    for i in range(3):
        spaces.append(hsv[:,:,i])
    for i in range(3):
        spaces.append(lab[:,:,i])
    for i in range(3):
        spaces.append(ycrcb[:,:,i])

    to_show = np.zeros(bgr.shape[:2])

    width_per = bgr.shape[1] // 4
    height_per = bgr.shape[0] // 3
    idx = 0
    y = 0
    for row in (1,2,3):
        x = 0
        for col in (1,2,3,4):
            to_show[y:y+height_per, x:x+width_per] = cv2.resize(spaces[idx], (width_per, height_per), cv2.INTER_CUBIC)
            idx += 1
            x += width_per
        y += height_per

    return show(to_show, time_ms=time_ms, fullscreen=True)


def distance(pt1, pt2):
    """
    Euclidean distance from pt1 to pt2
    """
    import math
    return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def Timed(fn):
    """
    Decorator to print duration of function calls
    """
    def wrapped(*args, **kwargs):
        timer = Timer()
        ret = fn(*args, **kwargs)
        print("%s took: %s" % (fn, timer))
        return ret
    return wrapped

# @Timed
def polygon_symmetric_diff(a, b):
    """
    Returns the symmetric difference between the two polygons. Ie. pixel count not shared by both polygons.
    NB: Probably assumes convex polygons
    """
    # Brute force approach. Make masks for each polygon and count the pixels in the symmetric difference
    x,y,w,h = cv2.boundingRect(np.concatenate((a,b)))
    a = a - (x,y)
    b = b - (x,y)
    a_mask = poly2mask(a, (h,w))
    b_mask = poly2mask(b, (h,w))
    sym_diff = cv2.bitwise_xor(a_mask, b_mask)
    return np.count_nonzero(sym_diff), (a_mask, b_mask)

def playground_score(known, detected):
    diff, masks = polygon_symmetric_diff(known, detected)
    known_area = masks[0]
    return 1.0 - diff/np.count_nonzero(known_area)

def matching_balls(known, detected, match_threshold_factor=1.0):
    """
    known: [(center, r), ...]
    detected:      [center, ...]
    match_threshold_factor: a ball x is considered a match of y if |x-y| < x_r*thresh

    We don't distinguish between ball types here.

    In this version we simply go through the known ball, matching them with the
    best match. No ball will be matched more than once.

    This doesn't necessarily maximize the score though:

    Should we attempt to find the optimal pairing? 

    Illustrative example: 
    *..O.*..O...*
    a  A b  B   c

    Assuming the radius of both A and B is 3 (...) should we attempt to pair B with b
    and A with a? Even though b match better with A.

    Or should b match with both A and B?
    Similarily: should A match with both a and b?
    """
    # There's only a few balls so we don't do anything fancy here yet

    matches = []
    unmatched = list(detected)

    for ball in known:
        center, r = ball
        match_idx, best_match = min(enumerate(unmatched), key=lambda x: distance(center, x[1]))
        d = distance(center, best_match)
        if d < r*match_threshold_factor:
            matches.append((ball, best_match))
            del unmatched[match_idx]

    return matches


def ball_detection_score(known, detected):
    # Three kinda of errors: (1) ball not detected at all, (2) offset from hand-detected center, (3) non-ball detected
    #
    # Only consider error 1 and 3 here
    #
    # Muliple possible scoring criteria:
    # - precision (same as accuracy in this case)
    # - recall
    # - F-score
    match_count = len(matching_balls(known, detected))

    recall = match_count / len(known) 
    precision = match_count / len(detected) 
    if recall + precision < 0.000000001:
        return 0, len(detected)
    # F-score, beta = 1:
    return 2*recall*precision / (recall + precision), len(detected)


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
