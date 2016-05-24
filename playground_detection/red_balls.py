#!/usr/bin/python3
import cv2
import numpy as np
import utilities
from image import Image
import logging
import os


logging.basicConfig() # omg..
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

from utilities import distance, power_threshold, make_debug_toggable, keypoint_filter_overlapping


show = make_debug_toggable(utilities.show, "red")


def red_ball_transform(image, exponent=1):
    if type(image) == Image:
        img = image.get_ycrcb(np.float32)
    else:
        # Assume image is a bgr 8bit ycrcb image (to please pipeline_visualizer..)
        img = image
        img = img.astype(np.float32)
        img *= 1./255

    # YCrCb color space should be suited for detect red objects. A whole channel just for our
    # purpose!
    Cr = img[:,:,1]

    return power_threshold(Cr, exponent)


def blob_detector(img, minArea=10, maxArea = 500, minDistBetweenBlobs = 100, minCircularity=0.7, maxCircularity=1.0, blobColor = 255):
    lightBlobParams = cv2.SimpleBlobDetector_Params()
    lightBlobParams.filterByArea = True
    lightBlobParams.minArea = minArea
    lightBlobParams.maxArea = maxArea
    lightBlobParams.minDistBetweenBlobs = minDistBetweenBlobs
    lightBlobParams.blobColor = blobColor
    lightBlobParams.filterByColor = True
    lightBlobParams.filterByConvexity = False
    lightBlobParams.filterByCircularity = True
    lightBlobParams.minCircularity = minCircularity
    lightBlobParams.maxCircularity = maxCircularity
    lightBlobDetector = cv2.SimpleBlobDetector_create(lightBlobParams)

    kpLight = lightBlobDetector.detect(img)
    return kpLight


def normalize_image(img):
    """
    Normalize a image so that the whole dynamic range (correct term?) is used.
    Assumes img is np.uint image.
    """
    img = img - np.min(img)
    img = img.astype(np.float32) / np.max(img)
    return np.round(img * 255).astype(np.uint8)

def threshold_rel(img, ratio, threshold_type=cv2.THRESH_BINARY):
    t = round(np.max(img)*ratio)
    return cv2.threshold(img, t, 255, threshold_type)[1]

## Works well, but sometimes find multiple hits per red ball
def surf_detector(img, hess_thresh=3000):
    surf = cv2.xfeatures2d.SURF_create(hess_thresh, upright=True) #, nOctaves=10)
    kps = surf.detect(img)

    # print("%s keypoints" % len(kps))
    kps = keypoint_filter_overlapping(kps)
    # print("\n".join(map(utilities.pretty_print_keypoint, kps)))
    # print("----------")

    return kps


class RedBallPlaygroundDetector:
    """
    Detecting the playground using four red balls in the corners of the field
    """
    def __init__(self, petanque_detection):
        self.petanque_detection = petanque_detection

    def detect(self, image):

        debug = "red" in os.environ.get("DEBUG", "")

        def blob():
            # Quick and dirty from pipeline_visualizer
            from pipeline_visualizer import apply_op, threshold_op, blob_op, params_marker_balls_blob as params

            show(image.bgr, scale=True)
            img = image.get_ycrcb(np.uint8)[:,:,1]
            show(img, scale=True)
            img = apply_op(img, threshold_op(), debug=debug, params=params)
            kps = blob_detector(img, **params["blob"])
            return kps

        def morph():
            from pipeline_visualizer import morph_open
            # Works well for rasberry west:
            # image.py no color normalization
            Cr = image.get_ycrcb(np.uint8)[:,:,1]
            # Cr = image.get_bgr(np.uint8)[:,:,2]
            temp = Cr
            show(temp, scale=True)
            temp = normalize_image(temp)
            show(temp, scale=True)
            temp = threshold_rel(Cr, 230/255) #, cv2.THRESH_TOZERO_INV)
            show(temp, scale=True)
            temp = morph_open(temp, 3, 4)
            # temp = cv2.blur(temp, (5,5))
            show(temp, scale=True)
            # temp = power_threshold(temp/255.0, 5)
            # show(temp, scale=True)
            kps = surf_detector(temp, hess_thresh=4000)
            # kps = sorted(kps, key=lambda kp: kp.response)[-4:]
            # print("\n".join(map(utilities.pretty_print_keypoint, kps)))
            show(temp, keypoints=kps, scale=True)
            return kps

        def surf():
            # Works well for rasberry west:
            # image.py no color normalization
            show(image.bgr, scale=True, text="RGB")
            Cr = image.get_ycrcb(np.uint8)[:,:,1]
            # Cr = image.get_bgr(np.uint8)[:,:,2]
            temp = Cr
            # show(temp, scale=True)
            # temp = normalize_image(temp)
            # show(temp, scale=True)
            # temp = threshold_rel(Cr, 188/255) #, cv2.THRESH_TOZERO_INV)
            # show(temp, scale=True)
            # temp = cv2.blur(temp, (5,5))
            show(temp, scale=True, text="Cr")
            temp = power_threshold(temp/255.0, 5)
            show(temp, scale=True, text="Compressed dynamic range")
            kps = surf_detector(temp, hess_thresh=4000)
            # kps = sorted(kps, key=lambda kp: kp.response)[-4:]
            # print("\n".join(map(utilities.pretty_print_keypoint, kps)))
            show(temp, keypoints=kps, scale=True, text="Detected points")
            return kps

        # kps = morph()
        kps = surf()
        # kps = blob()

        points = [(int(kp.pt[0]), int(kp.pt[1])) for kp in kps]

        if len(points) != 4:
            logger.debug("Found fewer than/more than 4 point (%s)", len(points))
            return points


        # Now we need to pair the detected points with the known real
        # world playground coordinates.
        #
        # We assume that the playground is a regular 2x6 rectangle and that the
        # camera angle is such that we can assume the longest lines in the
        # image corresponds to a long side in the real world.
        # NB: This doesn't hold for the rasberry west images :(
        #
        # We also assume we have exactly 4 detected points.
        #
        # Note that we don't actually need exact pairing since we don't care
        # what's up and what's down. Ie. as long as the orientation and
        # short/long linesegment ordering is the same we're good.
        #
        # The real-world coordintaes are given clockwise with
        # long-short-long-short line segment ordering.

        # First we find a ordering such that the resulting polygon has no crossing lines
        # .. and oriented correct.
        
        convex_hull = cv2.convexHull(np.array(points), clockwise=True)

        def cv_convex_hull_to_list(convex_hull):
            """
            cv2.convexHull wraps the points in extra lists for some reason.
            Unwrap and return as a python lists (of tuples) instead of a numpy array
            """
            convex_hull_as_list = []
            for idx in range(len(convex_hull)):
                vertex = convex_hull[idx][0]
                convex_hull_as_list.append((vertex[0], vertex[1]))
            return convex_hull_as_list

        points = cv_convex_hull_to_list(convex_hull)

        # Then we make sure the first line segment "long". Seems to be more robust to assume
        # the shortest segment actually is short (as opposed to the longest being "long"). At
        # least on the images we've taken so far.
        # a,b,c,d
        # a,b  b,c  c,d  d,a

        shortest_i, _ = min(enumerate(zip(points, points[1:]+points[0:1])),
                           key=lambda line_entry: utilities.distance(line_entry[1][0], line_entry[1][1]))



        # short-long-short-long
        points = points[shortest_i:] + points[:shortest_i]
        # rotate one to the right so we get a long segment first
        points = points[-1:] + points[:-1]

        return points

