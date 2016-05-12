#!/usr/bin/python3

from playground_detection.red_balls import RedBallPlaygroundDetector
from playground_detection.flood_fill import FloodFillPlaygroundDetector
from ball_detection.minimize_gradients import MinimizeGradientsBallDetector
from playground_detection.manual_playground_detector import ManualPlaygroundDetector
from ball_detection.hough import HoughBallDetector
from image import Image
import cv2
import utilities
import time
import numpy as np
import math
import os
import sys
import logging
from glob import glob

logging.basicConfig(stream=sys.stderr) # have to call this to get default console handler...
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

world_playground_polygon=((0,6000),(0,0),(2000,0),(2000,6000))

if __name__ == "__main__":

    filenames = []

    if len(sys.argv) > 1:
        filenames = sys.argv
    else:
        # for cur in os.walk(os.path.join(utilities.get_project_directory(), "images/microsoft_cam/24h/south/")):
        #     filenames = cur[2]
        #     break

        # filenames = glob("images/microsoft_cam/red_balls/*brightness=40,exposure_absolute=10,saturation=10.png")
        # filenames = glob("images/dual-lifecam,raspberry/raspberry/*.png")
        filenames = glob("images/dual-lifecam,raspberry/lifecam/*.png")

        filenames.sort()

    detector = RedBallPlaygroundDetector(None)

    try:
        for file in filenames:
            if not file.endswith(".png"):
                continue
            try:
                import datetime
                # date = datetime.datetime.strptime(file, "%Y-%m-%d_%H:%M:%S.png")
                # if date < datetime.datetime(2016, 4, 13, 7, 5):
                # if date < datetime.datetime(2016, 4, 12, 19, 0):
                #     continue
                image = Image(file, color_normalization=False)
                try:
                    camera_playground_polygon = detector.detect(image)
                except RuntimeError:
                    # print("Broken image: %s" % file)
                    print(file)
                    continue

                if "playground_poly" in image.get_metadata():
                    hand_detected = np.array(image.get_metadata()["playground_poly"])
                    score = utilities.playground_score(hand_detected, np.array(camera_playground_polygon))
                    logger.debug("Playground score: %s", score)
                    if score < 0.90:
                        # print("Bad image: %s" % file)
                        print(file)

            except FileNotFoundError:
                continue
            except ValueError:
                continue


    except Exception as e:
        import traceback
        print(e)
        traceback.print_tb(e.__traceback__)
    finally:
        if hasattr(detector, "transformer"):
            detector.transformer.save(filename="playground_transformer_state.json")
