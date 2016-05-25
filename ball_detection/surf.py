import math
import cv2
import numpy as np
import utilities
import logging
import os

from utilities import power_threshold, transform_image, as_uint8, make_debug_toggable


logging.basicConfig() # omg..
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


ball_transformation_params = {'ycrcb_cr': -0.75038994347347221, 'lab_b': 0.3036750425892179, 'bgr_b': -1.1892291465326323, 'lab_a': -0.7428254604861555, 'ycrcb_cn': -0.57301987482036387, 'lab_l': -0.68882136586824594, 'hsv_h': -0.095966576209467969, 'hsv_s': 0.45161636314988052, 'ycrcb_y': -0.21598565380357415, 'hsv_v': -1.3081319744285105, 'bgr_r': -1.2568352180214275, 'bgr_g': 0.49475196208293376}

def to_int(nested_tuple):
    if type(nested_tuple) in (list, tuple):
        return tuple(to_int(x) for x in nested_tuple)
    else:
        return int(nested_tuple)


def cv_unwrap(wrapped_points):
    """
    cv2.convexHull and cv2.findContours wraps the points in extra lists for some reason.
    """
    as_list = []
    for idx in range(len(wrapped_points)):
        vertex = wrapped_points[idx][0]
        as_list.append((vertex[0], vertex[1]))
    return np.array(as_list)

def circularity(ctr):
    perim = cv2.arcLength(ctr, closed=True)
    area = cv2.contourArea(ctr)
    if perim == 0:
        logger.debug("perimeter 0 in circularity calculation %s", ctr)
        return 0
    return (area*4*math.pi) / (perim*perim)

def eccentricity(ctr):
    ellipse = cv2.fitEllipse(ctr)
    (x,y),(w,h),alpha = ellipse
    return max(h,w)/min(h,w)

g_fbr_debug_counter = 0

class SurfBallDetector:
    def __init__(self, petanque_detection):
        self.petanque_detection = petanque_detection

    def detect(self, playground_image, w_H_p, playground_polygon, playground_mask):
        debug_spec = os.environ.get("DEBUG", "")
        show = make_debug_toggable(utilities.show, "surf")
        petanque_detection = self.petanque_detection

        def minimize_gradients(kps):
            from test import minimize_sum_of_squared_gradients

            optimized_kps = []
            tot = 0
            import math
            for kp in kps:
                kp_x = int(math.ceil(kp.pt[0]))
                kp_y = int(math.ceil(kp.pt[1]))
                expected_ball_radius = calc_playing_ball_radius(kp.pt)
                search_radius = int(math.ceil(2*expected_ball_radius))

                y_from = max(0, kp_y - search_radius)
                y_to = min(playground_image.get_bgr().shape[0], kp_y + search_radius)
                x_from = max(0, kp_x - search_radius)
                x_to = min(playground_image.get_bgr().shape[1], kp_x + search_radius)
                print(y_from, y_to, x_from, x_to, expected_ball_radius)

                possible_ball = playground_image.bgr[y_from:y_to, x_from:x_to]

                score, radius, dx, dy = minimize_sum_of_squared_gradients(possible_ball, expected_ball_radius)
                optimized_kps.append(cv2.KeyPoint(kp_x+dx, kp_y+dy, radius, score))
                # ball_matches.append((score, radius, kp_x+dx, kp_y+dy, kp_x, kp_y))
                tot += score

            show(img, keypoints=optimized_kps, scale=True, text="Minimized gradient optimized")
            return optimized_kps


        def detect_pig():
            """ Quick and dirty pig detection """

            from pipeline_visualizer import pig_pipeline, run_pipeline

            debug_pig = "pig" in debug_spec

            img = as_uint8(playground_image.bgr)
            img = run_pipeline(img, pig_pipeline, debug=debug_pig)

            _1, cv_ctrs, _2 = cv2.findContours(img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

            ctrs = [cv_unwrap(ctr) for ctr in cv_ctrs]

            # print(list(map(lambda ctr: eccentricity(ctr), ctrs)))
            # print(list(map(lambda ctr: circularity(ctr), ctrs)))

            ctrs = list(filter(lambda ctr: abs(1-circularity(ctr)) < 0.3, ctrs))

            circles = [cv2.minEnclosingCircle(ctr) for ctr in ctrs]

            if debug_pig:
                pig_debug_img = playground_image.bgr.copy()
                cv2.drawContours(pig_debug_img, cv_ctrs, -1, (0,0,255))
                for c in circles:
                    pt,r = to_int(c)
                    cv2.circle(pig_debug_img, pt, r, (0,255,0))
                    cv2.circle(pig_debug_img, pt, calc_pig_radius(pt), (255,0,0))
                show(pig_debug_img)

            if len(circles) == 0:
                return None

            pig_c, pig_r = min(circles, key=lambda c: abs(calc_pig_radius(c[0])-c[1]))
            return ((int(pig_c[0]), int(pig_c[1])), int(pig_r))


        def calc_pig_radius(pt):
            rw_radius = petanque_detection.pig_radius
            pg_radius = petanque_detection.get_playground_image_radius(pt, rw_radius, w_H_p)
            return pg_radius

        def calc_playing_ball_radius(pt):
            rw_radius = petanque_detection.ball_radius
            pg_radius = petanque_detection.get_playground_image_radius(pt, rw_radius, w_H_p)
            return pg_radius

        def filter_by_radius(kps):
            res = []
            for kp in kps:
                kp_radius = kp.size / 2
                pg_radius = calc_playing_ball_radius(kp.pt)
                radius_err = abs(pg_radius - kp_radius)
                if radius_err / pg_radius < 0.60:
                    res.append(kp)
                else:
                    logger.debug("kp (%.0f, %.0f) rejected error (rat) %.3f", kp.pt[0], kp.pt[1], radius_err / pg_radius)
                    if False:
                        global g_fbr_debug_counter
                        cv2.imwrite("debug/%d.png"%g_fbr_debug_counter, utilities.extract_circle(img, (kp.pt, kp.size/2), 40))
                        g_fbr_debug_counter += 1

            return res

        def filter_by_edges(kps):
            """
            Discards any keypoints intersecting the playground edges.
            (All keypoint-centers are assumed to be within the playground)
            """
            lines = list(zip(playground_polygon, playground_polygon[1:]+playground_polygon[:1]))
            res = []
            for kp in kps:
                min_dist_to_edge = min([utilities.distance_point_to_bounded_line(pt1, pt2, kp.pt) for (pt1, pt2) in lines])
                if min_dist_to_edge >= kp.size/2: # Size is diameter (ish)
                    res.append(kp)
                else:
                    pass

            return res

        def filter_pig_detected_as_playing_ball(kps, pig):
            if pig is None:
                return kps
            res = []
            for kp in kps:
                pig_r = pig[1]
                d = utilities.distance(kp.pt, pig[0])
                # Reject kps that overlap with the pig
                # A bit conservative? Maybe only reject kp if its center is within the pig radius?
                if kp.size/2+pig_r < d:
                    res.append(kp)

            return res

        img = transform_image(playground_image, ball_transformation_params)
        img = as_uint8(img) # SURF only works with uint8 images

        pig = detect_pig()
        if pig is None:
            logger.debug("Failed to detect pig")

        # img = cv2.cvtColor(playground_image.bgr, cv2.COLOR_BGR2GRAY)
        # img = as_uint8(img)

        img = cv2.blur(img, (7, 7))

        def surf_detector(img, hess_thresh=3000):
            surf = cv2.xfeatures2d.SURF_create(hess_thresh, upright=True) #, nOctaves=10)
            kps = surf.detect(img, mask=playground_mask)
            return kps

        kps = surf_detector(img)
        show(img, keypoints=kps, scale=True, text="Initial candidates")

        kps = utilities.keypoint_filter_overlapping(kps)
        show(img, keypoints=kps, scale=True, text="Overlaps removed")

        kps = filter_by_edges(kps)
        show(img, keypoints=kps, scale=True, text="Edge crossing removed")

        kps = filter_by_radius(kps)
        show(img, keypoints=kps, scale=True, text="Filtered by expected radius")

        kps = filter_pig_detected_as_playing_ball(kps, pig)

        kps = minimize_gradients(kps)
        show(img, keypoints=kps, scale=True, text="Gradient adjusted")


        if kps is not None:
            bgr = playground_image.get_bgr()
            ball_averages = []
            for kp in kps:
                expected_ball_radius = calc_playing_ball_radius(kp.pt)
                y_from = int(max(0, math.ceil(kp.pt[1]-expected_ball_radius)))
                y_to = int(min(bgr.shape[0], math.ceil(kp.pt[1]+expected_ball_radius)))
                x_from = int(min(bgr.shape[1], math.ceil(kp.pt[0]-expected_ball_radius)))
                x_to = int(max(0, math.ceil(kp.pt[0]+expected_ball_radius)))
                
                ball_averages.append(
                    (
                        kp
                    ,
                        np.average(bgr[y_from:y_to, x_from:x_to])
                    )
                )
            ball_averages.sort(key=lambda x: x[1])

            new_centers = [ball_averages[0][1], ball_averages[-1][1]]

            change = True
            while change:
                k_means = [(new_centers[0], []), (new_centers[1], [])]

                centers_at_start = [i[0] for i in k_means]
                for kp, avg in ball_averages:
                    closest = None
                    closest_dist = float("INF")
                    for island in k_means:
                        dist = abs(island[0] - avg)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest = island

                    if closest is not None:
                        closest[1].append((kp, avg))

                new_k_means = []
                for i in k_means:
                    kps = []
                    tot = 0.0
                    for kp, avg in i[1]:
                        kps.append((kp, avg))
                        tot += avg
                    if len(kps) != 0:
                        new_k_means.append((tot / len(kps), kps))
                    else:
                        new_k_means.append((i[0], []))

                new_centers = [i[0] for i in new_k_means]

                change = False
                for i in range(len(centers_at_start)):
                    if centers_at_start[i] != new_centers[i]:
                        change = True
                        break

            k_means.sort(key=lambda x: x[0])
            balls = []
            team_idx = 2
            for i in k_means:
                for kp, avg in i[1]:
                    balls.append(((int(kp.pt[0]), int(kp.pt[1])), team_idx))
                team_idx -= 1
            #balls = [((int(kp.pt[0]), int(kp.pt[1])), 1) for kp in kps]
        else:
            balls = [((0,0), 0)]

        if pig is not None:
            balls.append((pig[0], 0))
        else:
            balls[0] = ((balls[0][0]), 0) # arbitrary pig

        return balls
