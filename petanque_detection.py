#!/usr/bin/python3
from playground_detection.flood_fill import FloodFillPlaygroundDetector
from ball_detection.minimize_gradients import MinimizeGradientsBallDetector
from image import Image
import cv2
import utilities
import threading
import time


class PetanqueDetection:
    """
    All positions are width,height, except when used in numpy arrays
    All variables named image are of class Image
    The first point in playground_polygon is considered the origin
    """
    def __init__(self, playground_polygon=((0,0),(0,2),(2,6),(0,6)),
                 PlaygroundDetector=FloodFillPlaygroundDetector,
                 BallDetector=MinimizeGradientsBallDetector):

        self.playground_polygon = playground_polygon
        self.playground_detector = PlaygroundDetector(self)
        self.ball_detector = BallDetector(self)

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
        playground_polygon = self.playground_detector.detect(image)

        # Playground detection adjustment
        # Input: Original image, polygon defining the playground as list of points
        # Output: List of points of same size, defining the playground. First point is considered the origin.
        playground_polygon = self._user_adjust_playground(image, playground_polygon)

        # Ball detection
        # Input: Original image, polygon defining the playground as list of points
        # Output: List of tuples of two items. First item is the position (tuple:(x,y)), second item is the team number
        # the ball belongs to. Team 0 is reserved for the pig.
        balls = self.ball_detector.detect(image, playground_polygon)

        # Ball detection adjustment
        # Input: Original image, list of tuples of points where the balls are and the team they belong to
        # Output: List of points where the balls are, not necessarily the same size as the corresponding input
        balls = self._user_adjust_balls(image, balls)

        # Show result
        # Input: Original image, list of points where the balls are
        # Output: Balls and their ranks drawn onto and BGR image
        result = self._draw_distance_to_pig(image, balls)

        utilities.show(result)

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

    def _user_adjust_balls(self, image, balls):
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

                bgr = bgr_orig.copy()

                # Pig color, team 1 color, team 2 color
                ball_colors = [(200,0,0), (0,200,0), (0,0,200)]
                pressed_ball_colors = [(255,0,0), (0,255,0), (0,0,255)]

                pressed = balls[pressed_idx] if pressed_idx is not None else None
                mouseover = balls[mouseover_idx] if mouseover_idx is not None else None

                for idx, ball in enumerate(balls):
                    radius = 7
                    ball_position = ball[0]
                    ball_team = ball[1]
                    color = ball_colors[ball_team]
                    if ball in (mouseover, pressed):
                        radius = 11
                    if pressed == ball:
                        color = pressed_ball_colors[ball_team]
                    cv2.circle(bgr, ball_position, radius, color, cv2.FILLED)

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

    def _draw_distance_to_pig(self, image, balls):
        # TODO: Find homography from pixels to real world and measure the distance from each ball to the pig,
        # then draw a circle and number on top of all balls to indicate what position they are in
        return image.get_bgr()


if __name__ == "__main__":
    petanque_detection = PetanqueDetection()

    try:
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

    except Exception as e:
        import traceback
        print(e)
        traceback.print_tb(e.__traceback__)
