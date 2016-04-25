#!/usr/bin/python3
from playground_detection.flood_fill import FloodFillPlaygroundDetector
from ball_detection.minimize_gradients import MinimizeGradientsBallDetector
from image import Image
import cv2
import utilities
import threading
import time
import numpy as np
import math


class PetanqueDetection:
    """
    All positions are width,height, except when used in numpy arrays
    All variables named image are of class Image
    The first point in playground_polygon is considered the origin
    In this file, all homographies have the form to_H_from, where to and from are in (i, p, w), where:
        i = image (original image)
        p = playground_image
        w = real world
    Real world distance units are milli-meters
    """
    def __init__(self, playground_polygon=((0,6000),(0,0),(2000,0),(2000,6000)),
                 PlaygroundDetector=FloodFillPlaygroundDetector,
                 BallDetector=MinimizeGradientsBallDetector,
                 pig_radius=21.0,
                 ball_radius=35.8):

        self.real_world_playground_polygon = playground_polygon
        self.playground_detector = PlaygroundDetector(self)
        self.ball_detector = BallDetector(self)
        self.pig_radius = pig_radius
        self.ball_radius = ball_radius

        self._mouse_position = None
        self._win_name = "Petanque detector"
        self._window = cv2.namedWindow(self._win_name)

    def detect(self, image):
        """
        This is the main function used for detection. It is a pipeline where the inputs and outputs from the
        blocks are required to be of a specific format.
        """
        # Playground detection
        # Input: image: Image
        # Output: List of points of the same length as self.playground_polygon (even if it failed to detect anything!)
        # The first point is considered to be the origin, and going from point to the next in the list should trace
        # the convex hull of the points.
        camera_playground_polygon = self.playground_detector.detect(image)

        # Playground detection adjustment
        # Input: Original image, polygon defining the playground as list of points
        # Output: List of points of same size, defining the playground. First point is considered the origin.
        camera_playground_polygon = self._user_adjust_playground(image, camera_playground_polygon)

        # Homography estimation
        # Input: Camera playground polygon as a numpy array of points,
        # real world playground polygon as a numpy array of points
        # Output: numpy array with shape (3,3) that represents the homography matrix from the image to the real world
        w_H_i = self._find_homography_w_H_i(camera_playground_polygon)

        # Remove everything non-playground
        # Input: Original image, polygon defining the playground as list of points
        # Output: Image of the size of the bounding rectangle (not angled!) of the playground, with all pixels outside
        # the playground set to BGR (0,0,0)
        playground_image, w_H_p = self._get_playground_image(image, camera_playground_polygon, w_H_i)

        # Ball detection
        # Input: Image of the playground, homography from that image to the real world positions
        # Output: List of tuples of two items. First item is the position (tuple:(x,y)), second item is the team number
        # the ball belongs to. Team 0 is reserved for the pig. Returns coordinates of the balls in the playground_image!
        balls = self.ball_detector.detect(playground_image, w_H_p)

        # Ball detection adjustment
        # Input: Original image, list of tuples of points where the balls are and the team they belong to
        # Output: List of points where the balls are, not necessarily the same size as the corresponding input
        # Uses positions in the playground_image!
        balls = self._user_adjust_balls(playground_image, balls, w_H_p)

        # Show result
        # Input: Original image, list of points where the balls are in playground_image coordinates
        # Output: Balls and their ranks drawn onto and BGR image
        result = self._draw_distance_to_pig(image, balls, w_H_i, w_H_p)

        utilities.show(result, self._win_name)

    def _user_adjust_playground(self, image, playground_polygon):
        userdata = {}
        userdata["mouseover_idx"] = None
        userdata["pressed_idx"] = None
        userdata["polygon"] = playground_polygon
        userdata["bgr"] = image.get_bgr()
        userdata["lock"] = threading.Lock()
        userdata["run"] = True

        def draw_polygon(userdata):
            with userdata["lock"]:
                bgr_orig = userdata["bgr"].copy()

            while True:
                start_s = time.time()

                with userdata["lock"]:
                    if not userdata["run"]:
                        return

                    polygon = userdata["polygon"].copy()
                    mouseover_idx = userdata["mouseover_idx"]
                    pressed_idx = userdata["pressed_idx"]

                bgr = bgr_orig.copy()

                line_color = (0, 0, 255)
                dot_color = (255, 0, 0)
                pressed_dot_color = (150, 0, 0)
                origin_dot_color = (0, 255, 0)
                mouseover = polygon[mouseover_idx] if mouseover_idx is not None else None
                pressed = polygon[pressed_idx] if pressed_idx is not None else None

                for idx, pt1 in enumerate(polygon):
                    pt2 = polygon[(idx+1)%len(polygon)]

                    cv2.line(bgr, pt1, pt2, line_color, 3)

                for idx, pt in enumerate(polygon):
                    radius = 7
                    color = dot_color
                    if pt in (mouseover, pressed):
                        radius = 11
                    if pressed == pt:
                        color = pressed_dot_color
                    if idx == 0:
                        color = origin_dot_color
                    cv2.circle(bgr, pt, radius, color, cv2.FILLED)

                cv2.imshow(self._win_name, bgr)
                end_s = time.time()
                time_used_ms = (end_s - start_s) * 1000
                # Need to call waitKey to avoid weird artifacts, and the minimum we can wait is 1 ms (0 means forever)
                time_to_wait_ms = int(max(1, 1000/60 - time_used_ms))
                cv2.waitKey(time_to_wait_ms)

        def mouse_event(event, x, y, a, userdata):
            with userdata["lock"]:
                polygon = userdata["polygon"].copy()
                pressed_idx = userdata["pressed_idx"]
                mouseover_idx = userdata["mouseover_idx"]

            if event == cv2.EVENT_LBUTTONUP:
                pressed_idx = None

            elif event == cv2.EVENT_MOUSEMOVE:
                if pressed_idx is not None:
                    polygon[pressed_idx] = (x, y)

                else:
                    mouseover_idx = None
                    for idx, pt in enumerate(polygon):
                        if utilities.distance(pt, (x, y)) < 10:
                            mouseover_idx = idx
                            break

            elif event == cv2.EVENT_LBUTTONDOWN:
                for idx, pt in enumerate(polygon):
                    if utilities.distance(pt, (x, y)) < 10:
                        pressed_idx = idx
                        break

            with userdata["lock"]:
                userdata["pressed_idx"] = pressed_idx
                userdata["mouseover_idx"] = mouseover_idx
                userdata["polygon"] = polygon

        render_thread = threading.Thread(target=draw_polygon, args=(userdata,))
        render_thread.start()
        cv2.setMouseCallback(self._win_name, mouse_event, userdata)

        run = True
        while run:
            key = utilities.wait_for_key()

            with userdata["lock"]:
                mouseover_idx = userdata["mouseover_idx"]
                pressed_idx = userdata["pressed_idx"]
                polygon = userdata["polygon"].copy()

            if key in ('s', 'q'):
                run = False

            elif key == 'o':
                if mouseover_idx is not None:
                    polygon = polygon[mouseover_idx:] + polygon[:mouseover_idx]
                    mouseover_idx = 0
                    pressed_idx = 0 if pressed_idx is not None else None

            with userdata["lock"]:
                userdata["run"] = run
                userdata["polygon"] = polygon
                userdata["mouseover_idx"] = mouseover_idx
                userdata["pressed_idx"] = pressed_idx

        render_thread.join()

        if key == 'q':
            exit(0)

        return userdata["polygon"]

    def _user_adjust_balls(self, image, balls, w_H_p):
        p_H_w = np.linalg.inv(w_H_p)
        userdata = {}
        userdata["pressed_idx"] = None
        userdata["mouseover_idx"] = None
        userdata["balls"] = balls
        userdata["bgr"] = image.get_bgr()
        userdata["cur_mouse_pos"] = (0,0)
        userdata["lock"] = threading.Lock()
        userdata["run"] = True

        def draw_balls(userdata):
            with userdata["lock"]:
                bgr_orig = userdata["bgr"].copy()

            while True:
                start_s = time.time()

                with userdata["lock"]:
                    if not userdata["run"]:
                        return

                    balls = userdata["balls"].copy()
                    pressed_idx = userdata["pressed_idx"]
                    mouseover_idx = userdata["mouseover_idx"]
                    cur_mouse_pos = userdata["cur_mouse_pos"]

                bgr = bgr_orig.copy()

                # Pig color, team 1 color, team 2 color
                ball_colors = [(200,0,0), (0,200,0), (0,0,200)]
                pressed_ball_colors = [(255,0,0), (0,255,0), (0,0,255)]

                pressed = balls[pressed_idx] if pressed_idx is not None else None
                mouseover = balls[mouseover_idx] if mouseover_idx is not None else None

                def get_playground_image_radius(playground_position, real_world_radius):
                    world_pos = np.dot(w_H_p, np.array([playground_position[0], playground_position[1], 1]).reshape((3,1)))
                    world_pos /= world_pos[2][0]
                    d_pos = np.dot(w_H_p, np.array([playground_position[0]+10000, playground_position[1], 1]).reshape((3,1)))
                    d_pos /= d_pos[2][0]
                    delta_world = d_pos - world_pos
                    x, y = delta_world[0][0], delta_world[1][0]
                    # TODO: Check for x == 0
                    a = y / x

                    new_x = math.sqrt(real_world_radius**2/(a**2 + 1))
                    new_y = a*new_x
                    new_x += world_pos[0][0]
                    new_y += world_pos[1][0]
                    new_world_pos = np.array([new_x, new_y, 1]).reshape((3,1))
                    new_screen_pos = np.dot(p_H_w, new_world_pos)
                    new_screen_pos /= new_screen_pos[2][0]

                    cur_world_pos_numpy = np.array([playground_position[0], playground_position[1], 1]).reshape((3,1))
                    diff = new_screen_pos - cur_world_pos_numpy
                    return int(round(math.sqrt(diff[0][0]**2 + diff[1][0]**2)))

                for idx, ball in enumerate(balls):
                    ball_position = ball[0]
                    ball_team = ball[1]
                    color = ball_colors[ball_team]
                    ball_real_world_radius = self.pig_radius if ball_team == 0 else self.ball_radius
                    radius = get_playground_image_radius(ball_position, ball_real_world_radius)

                    if ball in (mouseover, pressed):
                        radius *= 1.3
                        radius = int(round(radius))
                    if pressed == ball:
                        color = pressed_ball_colors[ball_team]
                    cv2.circle(bgr, ball_position, radius, color, thickness=2)

                cv2.imshow(self._win_name, bgr)

                end_s = time.time()
                time_used_ms = (end_s - start_s) * 1000
                # Need to call waitKey to avoid weird artifacts, and the minimum we can wait is 1 ms (0 means forever)
                time_to_wait_ms = int(max(1, 1000/60 - time_used_ms))
                cv2.waitKey(time_to_wait_ms)

        def mouse_event(event, x, y, a, userdata):
            with userdata["lock"]:
                balls = userdata["balls"].copy()
                pressed_idx = userdata["pressed_idx"]
                mouseover_idx = userdata["mouseover_idx"]

            cur_mouse_pos = (x,y)

            if event == cv2.EVENT_LBUTTONUP:
                pressed_idx = None

            elif event == cv2.EVENT_MOUSEMOVE:
                if pressed_idx is not None:
                    _, ball_team = balls[pressed_idx]
                    balls[pressed_idx] = ((x, y), ball_team)

                else:
                    mouseover_idx = None
                    for idx, (ball_pos, ball_team) in enumerate(balls):
                        if utilities.distance(ball_pos, (x, y)) < 10:
                            mouseover_idx = idx
                            break

            elif event == cv2.EVENT_LBUTTONDOWN:
                if mouseover_idx is not None:
                    pressed_idx = mouseover_idx

            with userdata["lock"]:
                userdata["pressed_idx"] = pressed_idx
                userdata["mouseover_idx"] = mouseover_idx
                userdata["balls"] = balls
                userdata["cur_mouse_pos"] = cur_mouse_pos

        def get_num_pigs(balls):
            return len([i for i in balls if i[1] == 0])

        render_thread = threading.Thread(target=draw_balls, args=(userdata,))
        render_thread.start()
        cv2.setMouseCallback(self._win_name, mouse_event, userdata)

        run = True
        while run:
            key = utilities.wait_for_key()

            with userdata["lock"]:
                pressed_idx = userdata["pressed_idx"]
                mouseover_idx = userdata["mouseover_idx"]
                balls = userdata["balls"].copy()
                cur_mouse_pos = userdata["cur_mouse_pos"]

            mouseover = balls[mouseover_idx] if mouseover_idx is not None else None
            pressed = balls[pressed_idx] if pressed_idx is not None else None

            if key in ('s', 'q'):
                run = False

            elif key == 'a':
                balls.append((cur_mouse_pos, 1))
                mouseover_idx = len(balls) - 1
                pressed_idx = len(balls) - 1 if pressed_idx is not None else None

            elif key == 'd':
                if mouseover is not None:
                    balls.remove(mouseover)
                    mouseover_idx = None
                    pressed_idx = None

            elif key in ('0', '1', '2'):
                if mouseover is not None:
                    team_num = int(key)
                    if team_num != 0 or get_num_pigs(balls) == 0:
                        balls[mouseover_idx] = (mouseover[0], team_num)

            with userdata["lock"]:
                userdata["pressed_idx"] = pressed_idx
                userdata["mouseover_idx"] = mouseover_idx
                userdata["balls"] = balls
                userdata["run"] = run

        render_thread.join()

        if key == 'q':
            exit(0)

        return userdata["balls"]

    def _find_homography_w_H_i(self, camera_playground_polygon):
        w_H_i = cv2.findHomography(np.array(camera_playground_polygon),
                                   np.array(self.real_world_playground_polygon))[0]

        return w_H_i
        utilities.show(cv2.warpPerspective(image.get_bgr(), w_H_i, (200,600)))

    def _get_playground_image(self, image, camera_playground_polygon, w_H_i):
        # Get the bounding rect so we can resize the image down to that size
        x, y, w, h = cv2.boundingRect(np.array(camera_playground_polygon))
        image_bgr = image.get_bgr()
        playground_bgr = image_bgr[y:y+h, x:x+w, :].copy()
        playground_mask = utilities.poly2mask(camera_playground_polygon, image_bgr)[y:y+h, x:x+w]
        playground_bgr[np.where(playground_mask == 0)] = 0

        # First make the homography from the playground_image to the image (simple translation)
        i_H_p = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])
        # Then multiply them together to get the homography from the playground_image to real world
        w_H_p = np.dot(w_H_i, i_H_p)

        return Image(image_data=playground_bgr, color_normalization=False), w_H_p

    def _draw_distance_to_pig(self, image, balls,
                              w_H_i,
                              playground_image_to_real_world_homography):
        """
        balls should have the coordinates in playground_image coordinates
        """
        to_show = image.get_bgr(np.uint8).copy()

        pig_position = None
        balls_real_world = []
        for (x, y), team in balls:
            real_position = np.dot(playground_image_to_real_world_homography,
                                   np.array([x, y, 1]).reshape((3,1)))
            real_position /= real_position[2][0]

            # Team 0 is the pig
            if team == 0:
                pig_position = (real_position[0][0], real_position[1][0])
            else:
                balls_real_world.append(((real_position[0][0], real_position[1][0]), team))

        # Sort the balls by their distance (in the real world) to the pig
        balls_real_world.sort(key=lambda x: utilities.distance(pig_position, x[0]))

        i_H_w = np.linalg.inv(w_H_i)

        rank = 1
        for (x_w, y_w), team in balls_real_world:
            i_pos = np.dot(i_H_w, np.array([x_w, y_w, 1]).reshape((3,1)))
            i_pos /= i_pos[2][0]
            x = int(round(i_pos[0][0]))
            y = int(round(i_pos[1][0]))
            padding = 2
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 1
            text = str(rank)
            text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
            text_size = (int(round(text_size[0])), int(round(text_size[1])))

            cv2.rectangle(to_show, (x-padding, y-text_size[1]-padding), (x+text_size[0]+padding, y+padding), (0, 0, 0), cv2.FILLED)
            cv2.putText(to_show, text, (x, y), font_face, font_scale, (255, 255, 255), thickness)

            rank += 1
        return to_show


if __name__ == "__main__":
    petanque_detection = PetanqueDetection()

    # try:
    import os
    filenames = []
    for cur in os.walk(os.path.join(utilities.get_project_directory(), "images/microsoft_cam/24h/south/")):
        filenames = cur[2]
        break

    filenames.sort()

    for file in filenames:
        try:
            import datetime
            date = datetime.datetime.strptime(file, "%Y-%m-%d_%H:%M:%S.png")
            # if date < datetime.datetime(2016, 4, 13, 7, 5):
            # if date < datetime.datetime(2016, 4, 12, 19, 0):
            #     continue
            image = Image(os.path.join(utilities.get_project_directory(), "images/microsoft_cam/24h/south/", file))
        except FileNotFoundError:
            continue

        petanque_detection.detect(image)

    # except Exception as e:
    #     import traceback
    #     print(e)
    #     traceback.print_tb(e.__traceback__)
