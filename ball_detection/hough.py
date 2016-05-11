import cv2
import numpy as np


def hough(img, min_radius, max_radius, p1, p2):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=p1, param2=p2, minRadius=min_radius, maxRadius=max_radius)
    return circles

class HoughBallDetector:
    def __init__(self, petanque_detection):
        self.petanque_detection = petanque_detection

    def detect(self, playground_image, w_H_p):
        params = {
            "blur": {'size': 11},
            "hough": {'p1': 57, 'p2': 8, 'min_radius': 5, 'max_radius': 14},
        }
        gray = playground_image.get_gray(np.uint8)

        blur_size = params["blur"]["size"]
        gray = cv2.blur(gray, (blur_size, blur_size))
        circles = hough(gray, **params["hough"])

        if circles is not None:
            balls = [((c[0], c[1]), 1) for c in circles[0,:].astype(np.int32)]
            balls[0] = ((balls[0][0]), 0) # arbitrary pig
        else:
            balls = [((0,0), 0)]

        return balls
