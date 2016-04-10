#!/usr/bin/python3
import cv2
import numpy as np
import math


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


def flood_fill_until(img, limit, center=None, color=255, max_diff=25):
    """
    Increase the loDiff and upDiff incrementally until limit of the pixels in the image has been filled
    """
    seed = center
    if seed is None:
        seed = get_middle(img)

    diff = 0
    num_filled = 0
    mask_shape = (img.shape[0]+2, img.shape[1]+2)
    while num_filled < limit*img.size and diff < max_diff:
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
