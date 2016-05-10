#!/usr/bin/python3
import cv2
import numpy as np
import utilities
from image import Image


def red_ball_transform(image, exponent=1):
    if type(image) == Image:
        img = image.get_ycrcb()
    else:
        img = image
        img = img.astype(np.float32)
        img *= 1./255

    Y = img[:,:,1]

    Yp = np.power(Y, exponent)

    amax = np.amax(Yp)
    light = Yp / amax
    light *= 255
    light = np.clip(light, 0, 255)
    light = light.astype(np.uint8)

    return light

    # return img[:,:,1]
    # utilities.show(img[:,:,1])
    # img = img.astype(np.float32)
    # img = img[:,:,1] + img[:,:,0]
    # utilities.show(img)
    # print(img.shape)
    # img = np.round(img / np.max(img))
    # img = img.astype(np.uint8)
    # return img

def blob_detector(img, minArea=10, maxArea = 500, minDistBetweenBlobs = 100, blobColor = 255):

    # Use hue and saturation
    lightBlobParams = cv2.SimpleBlobDetector_Params()
    lightBlobParams.filterByArea = True
    lightBlobParams.minArea = minArea
    lightBlobParams.maxArea = maxArea
    lightBlobParams.minDistBetweenBlobs = minDistBetweenBlobs
    lightBlobParams.blobColor = blobColor
    lightBlobParams.filterByColor = False
    lightBlobParams.filterByConvexity = False
    lightBlobDetector = cv2.SimpleBlobDetector_create(lightBlobParams)

    # out = np.array((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    kpLight = lightBlobDetector.detect(img)
    # out = cv2.drawKeypoints(img, kpLight, None, color=[0, 0, 255])
    return kpLight

## Works well, but sometimes find multiple hits per red ball
def surf_detector(img):
    surf = cv2.xfeatures2d.SURF_create(2500, upright=True)
    kps = surf.detect(img)
    return kps

class RedBallPlaygroundDetector:
    """
    Detecting the playground using four red balls in the corners of the field
    """
    def __init__(self, petanque_detection):
        self.petanque_detection = petanque_detection

    def detect(self, image):
        # Quick and dirty from pipeline_visualizer
        params = {
            "trans": {'exponent': 5},
            "blob": {'minArea': 15, 'minDistBetweenBlobs': 120, 'blobColor': 255, 'maxArea': 208},
        }
        img = red_ball_transform(image, **params["trans"])
        kps = blob_detector(img, **params["blob"])

        points = [(int(kp.pt[0]), int(kp.pt[1])) for kp in kps]


        # Now we need to pair the detected points with the known real
        # world playground coordinates.
        #
        # We assume that the playground is a regular 2x6 rectangle and that the
        # camera angle is such that we can assume the longest lines in the
        # image corresponds to a long side in the real world.
        #
        # Note that we don't actually need exact pairing since we don't care
        # what's up and what's down. Ie. as long as the orientation and
        # short/long linesegment ordering is the same we're good.
        #
        # The real-world coordintaes are given clockwise long-short-long-short

        # First we find a ordering such that the resulting polygon has no crossing lines.
        # .. and oriented correct.
        
        convex_hull = cv2.convexHull(np.array(points), clockwise=True)

        def cv_convex_hull_to_list(convex_hull):
            """
            cv2.convectHull wraps the points in extra lists for some reason.
            Unwrap and return as a python lists (of tuples) instead of a numpy array
            """
            convex_hull_as_list = []
            for idx in range(len(convex_hull)):
                vertex = convex_hull[idx][0]
                convex_hull_as_list.append((vertex[0], vertex[1]))
            return convex_hull_as_list

        points = cv_convex_hull_to_list(convex_hull)

        # Then we make sure the first line segment is the longest:
        # a,b,c,d
        # a,b  b,c  c,d  d,a
        longest_i, _ = max(enumerate(zip(points, points[1:]+points[0:1])),
                           key=lambda line_entry: utilities.distance(line_entry[1][0], line_entry[1][1]))

        points = points[longest_i:] + points[:longest_i]

        return points


        
