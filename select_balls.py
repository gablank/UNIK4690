
import numpy as np
import cv2
import utilities
from utilities import read_metadata
from utilities import update_metadata
import sys               
import os                
from glob import glob    
import math
from image import Image
                         
from utilities import two_point_rect_to_bb

def extract_circle(img, circle, margin=0, mask_color=None):
    (cx,cy), r = circle
    x,y = cx-r-margin, cy-y-margin
    w = r*2 + margin*2
    h = w
    roi = img[y:y+h, x:x+w]
    if mask_color:
        mask = np.ones((h, w), dtype=np.uint8)
        cv2.circle(mask, circle[0], r, 255, -1)
        roi[np.where(mask == 0)] = mask_color
    return roi
    

if __name__ == '__main__':
    path = "images/microsoft_cam/24h/south/2016-04-12_16:19:04.png"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    metadata = read_metadata(path)
    if "ball_bbs" not in metadata:
        img = cv2.imread(path)
        ball_bbs = utilities.select_rects(img)
        metadata = update_metadata(os.path.dirname(path), {"ball_bbs": ball_bbs})

    if "ball_circles" not in metadata:
        img = cv2.imread(path)
        circles_bbs = utilities.select_circles(img)
        metadata = update_metadata(os.path.dirname(path), {"ball_circles": circles_bbs})

    filenames = glob(os.path.join(utilities.get_project_directory(), "images/microsoft_cam/24h/south/*.png"))
    filenames.sort()

    ball_bbs = ([two_point_rect_to_bb(*bb) for bb in metadata["ball_bbs"]])
    ball_bbs = sorted(ball_bbs, key=lambda bb: bb[2:])
    ball_bbs = np.array(ball_bbs)

    ball_count = len(ball_bbs)

    border = 2
    cols = ball_count
    rows = math.ceil(ball_count / cols)

    req_ball_w = sum(ball_bbs [:,3]) # inexact, but safe and easy
    req_ball_h = sum(ball_bbs [:,2]) # inexact, but safe and easy

    canvas = np.zeros(((req_ball_h+border)*rows, req_ball_w+ball_count*border, 3),
                      dtype=np.float32)

    scale = 3
    target_size = (canvas.shape[1]*scale, canvas.shape[0]*scale)

    for file_path in filenames:
        img = Image(file_path)

        pix = img.get_bgr()

        tx, ty = 0, 0
        col = 0
        for i, bb in enumerate(ball_bbs):
            x, y, w, h = bb
            canvas[ty:ty+h, tx:tx+w] = pix[y:y+h, x:x+w]
            tx += w+border
            col += 1
            if col == cols:
                col = 0
                tx = 0
                ty += req_ball_h + border

        frame = canvas if scale == 1 else cv2.resize(canvas, target_size)


        utilities.show(frame, time_ms=30)
    


