#!/usr/bin/env python3

from playground_detection.red_balls import RedBallPlaygroundDetector
from playground_detection.flood_fill import FloodFillPlaygroundDetector
from ball_detection.minimize_gradients import MinimizeGradientsBallDetector
from playground_detection.manual_playground_detector import ManualPlaygroundDetector
from ball_detection.hough import HoughBallDetector
from ball_detection.surf import SurfBallDetector
from petanque_detection import PetanqueDetection
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

    # petanque_detection = PetanqueDetection()
    # petanque_detection = PetanqueDetection(PlaygroundDetector=ManualPlaygroundDetector,
    #                                        BallDetector=HoughBallDetector)
    # petanque_detection = PetanqueDetection(PlaygroundDetector=RedBallPlaygroundDetector,
    #                                        BallDetector=HoughBallDetector)
    petanque_detection = PetanqueDetection(PlaygroundDetector=RedBallPlaygroundDetector,
                                           BallDetector=SurfBallDetector)
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
                image = Image(file, histogram_equalization=None)
                try:
                    result = petanque_detection.detect(image, interactive=False)
                    _, pg_score, ball_score, ball_count, real_count = result
                    logger.info("Result: pgs: % .2f, bs: %5.2f, bd: % d",
                                 pg_score, ball_count, real_count-ball_count)
                    if pg_score < 0.8:
                        print("pg: %s" % file)
                    if ball_count != real_count:
                        print("b: %s" % file)
                except Exception as e:
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.error("Error: %s", e)

            except FileNotFoundError:
                continue
            except ValueError:
                continue


    except Exception as e:
        import traceback
        print(e)
        traceback.print_tb(e.__traceback__)
    finally:
        if hasattr(petanque_detection.playground_detector, "transformer"):
            petanque_detection.playground_detector.transformer.save(filename="playground_transformer_state.json")
