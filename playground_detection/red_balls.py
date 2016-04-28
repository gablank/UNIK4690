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
    out = cv2.drawKeypoints(img, kpLight, None, color=[0, 0, 255])
    return out, kpLight

class RedBallPlaygroundDetector:
    """
    Detecting the playground using four red balls in the corners of the field
    """
    def __init__(self, petanque_detection):
        self.petanque_detection = petanque_detection

    def detect(self, image):
        params = {
            "trans": {'exponent': 4.87},
            "blob": {'minArea': 15, 'minDistBetweenBlobs': 120, 'blobColor': 255, 'maxArea': 208},
        }
        img = red_ball_transform(image, **params["trans"])
        _, kps = blob_detector(img, **params["blob"])
        return [(int(kp.pt[0]), int(kp.pt[1])) for kp in kps]
