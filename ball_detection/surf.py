import cv2
import numpy as np
import utilities

from utilities import power_threshold, transform_image, as_uint8, make_debug_toggable


ball_transformation_params = {'ycrcb_cr': -0.75038994347347221, 'lab_b': 0.3036750425892179, 'bgr_b': -1.1892291465326323, 'lab_a': -0.7428254604861555, 'ycrcb_cn': -0.57301987482036387, 'lab_l': -0.68882136586824594, 'hsv_h': -0.095966576209467969, 'hsv_s': 0.45161636314988052, 'ycrcb_y': -0.21598565380357415, 'hsv_v': -1.3081319744285105, 'bgr_r': -1.2568352180214275, 'bgr_g': 0.49475196208293376}

def surf_detector(img, hess_thresh=3000):
    surf = cv2.xfeatures2d.SURF_create(hess_thresh, upright=True) #, nOctaves=10)
    kps = surf.detect(img)
    return kps

class SurfBallDetector:
    def __init__(self, petanque_detection):
        self.petanque_detection = petanque_detection

    def detect(self, playground_image, w_H_p):
        show = make_debug_toggable(utilities.show, "surf")
        petanque_detection = self.petanque_detection

        def calc_pg_radius(kp):
            rw_radius = petanque_detection.ball_radius
            pg_radius = petanque_detection.get_playground_image_radius(kp.pt, rw_radius, w_H_p)

            return pg_radius

        def filter_by_radius(kps):
            res = []
            for kp in kps:
                pg_radius = calc_pg_radius(kp)
                radius_err = abs(pg_radius - kp.size)
                print(pg_radius, kp.size, radius_err / pg_radius)
                if radius_err / pg_radius < 0.5:
                    res.append(kp)

            return res

        img = transform_image(playground_image, ball_transformation_params)
        img = as_uint8(img)

        img = cv2.blur(img, (7, 7))

        kps = surf_detector(img)
       
        show(img, keypoints=kps, scale=True)

        # n = len(kps)
        kps = utilities.keypoint_filter_overlapping(kps)
        # print("filtered %s" % (n-len(kps)))
        show(img, keypoints=kps, scale=True)

        # n = len(kps)
        # kps = filter_by_radius(kps)
        # print(kps)
        # show(img, keypoints=kps, scale=True)
        # print("filtered %s" % (n-len(kps)))


        # points = [(int(kp.pt[0]), int(kp.pt[1])) for kp in kps]


        if kps is not None:
            balls = [((int(kp.pt[0]), int(kp.pt[1])), 1) for kp in kps]
            balls[0] = ((balls[0][0]), 0) # arbitrary pig
        else:
            balls = [((0,0), 0)]

        return balls
