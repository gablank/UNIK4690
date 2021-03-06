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
import json
import argparse

logging.basicConfig(stream=sys.stderr) # have to call this to get default console handler...
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":

    playground_only = False

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--playground-only', action='store_true',
                        help='Only detect playground')

        args, filenames = parser.parse_known_args()
        playground_only = args.playground_only
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

    pg_thresh = 0.8

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
                image = Image(file, histogram_equalization=None, undistort=True, flip=False)
                # image = Image(file, histogram_equalization=None)
                try:
                    result = petanque_detection.detect(image, interactive=False, playground_only=playground_only)
                    _, pg_score, ball_score, ball_count, real_count, piglet_score, offset_error_sum = result
                    logger.info("Result: pgs: % .2f, bs: %5.2f, bd: % d, piglet: %d, offset_error: %.2f" ,
                                 pg_score, ball_count, real_count-ball_count, piglet_score, offset_error_sum)
                    if pg_score < pg_thresh:
                        print("pg: %s" % file)
                    if ball_count != real_count:
                        print("b: %s" % file)
                    if piglet_score == 0:
                        print("p: %s" % file)
                except Exception as e:
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.error("Error: %s", e)

            except FileNotFoundError:
                continue
            except ValueError:
                continue

        def iter_count(it):
            return sum(1 for i in it)

        stats = petanque_detection.statistics
        n = len(stats)
        # img, pg, bs, b, ab, p, delta sum 
        failed_pg = iter_count(filter(lambda x: x[1] < pg_thresh, stats))
        failed_balls = iter_count(filter(lambda x: x[3] != x[4], stats))
        ball_delta_abs_sum = sum(map(lambda x: abs(x[4]-x[3]), stats))
        failed_piglets = sum(map(lambda x: 1-x[-2], stats))
        avg_distance_error = sum(map(lambda x: x[-1]/x[3], stats))
        logger.info("Playground failed: %.2f", failed_pg/n)
        logger.info("Balls failed     : %.2f", failed_balls/n)
        logger.info("Ball error sum     : %d", ball_delta_abs_sum)
        logger.info("Ball error sum rat : %.2f", ball_delta_abs_sum/(7*n)) # NB! assume 7 balls
        logger.info("Piglets failed     : %.2f", failed_piglets/n)
        logger.info("Average dist err   : %.2f", avg_distance_error/n)


        with open("stats.json", "w") as fp:
            json.dump(stats, fp) # overwrites on error too... 


    except Exception as e:
        import traceback
        print(e)
        traceback.print_tb(e.__traceback__)
    finally:
        if hasattr(petanque_detection.playground_detector, "transformer"):
            petanque_detection.playground_detector.transformer.save(filename="playground_transformer_state.json")
