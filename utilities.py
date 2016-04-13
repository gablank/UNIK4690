#!/usr/bin/python3
import cv2
import numpy as np
import math
import json
import os


def get_middle(img):
    """
    Get the middle of the image
    Res: P: (x,y)
    """
    mid_x = int(round(img.shape[1] / 2, 0))
    mid_y = int(round(img.shape[0] / 2, 0))
    return (mid_x, mid_y)


def show(img, win_name="test", fullscreen=False, time_ms=0):
    """
    Show img in a window
    """
    if fullscreen:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow(win_name, img)

    key = cv2.waitKey(time_ms)
    if key % 256 == ord('q'):
        exit(0)

    cv2.destroyWindow(win_name)


def box(img, center, side):
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


def wait_for_key(char):
    while(True):
        key = cv2.waitKey()
        # http://stackoverflow.com/a/17284668/1517969
        if (key % 256) == ord('s'):
            break

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
    wait_for_key("s")
    cv2.destroyWindow("polygon-select")
    return polygon

from matplotlib import pyplot as plt
def plot_histogram(img, channels=[0], mask=None, color="b", max=None):
    """
    Adds the histogram to the active matplotlib plot. Use plt.show() after to show the plot.
    """
    max = np.max(img)+0.00001 if max is None else max
    for ch in channels:
        hist = cv2.calcHist([img], [ch], mask, [256], [0, max])
        hist = hist/sum(hist) # normalize so each bucket represents percentage of total pixels
        plt.plot(hist, color)

def get_metadata_path(img_path):
    img_name = os.path.basename(img_path)
    img_base = "".join(img_name.split(".")[:-1])
    img_dir = os.path.dirname(img_path)
    series_metadata_path = os.path.join(img_dir, "metadata.json")
    if os.path.exists(series_metadata_path):
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
        json.dump(meta_dict, fp)
    return meta_dict

if __name__ == "__main__":
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
