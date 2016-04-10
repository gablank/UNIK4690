#!/usr/bin/python3
import utilities


class PlaygroundDetectionBase:
    def __init__(self, img):
        self.img = img
        self.working_copy = None
        self.mid = utilities.get_middle(img)
        self.mid_x = self.mid[0]
        self.mid_y = self.mid[1]

    def detect(self):
        pass

    def _get_convex_hull(self):
        pass

    def _get_square(self, convex_hull):
        pass
