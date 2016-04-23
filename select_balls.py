
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

# Might be some one off errors in pixel handling here

def test_circle_drawing():
    canvas = np.zeros((20,20,3))
    canvas[10,10] = (0,0,255)
    cv2.circle(canvas, (10,10), 4, (255,0,0))
    cv2.namedWindow("win")
    cv2.imshow("win", cv2.resize(canvas, (80, 80)))
    cv2.waitKey()
    cv2.destroyAllWindows()

def circle_bb(circle):
    (cx,cy), r = circle
    x, y = cx-r, cy-r

    # How to interpret radius and center in a pixel world?
    # Kinda makes sense to to force odd diameter, but seems like opencv don't do that.
    # https://github.com/Itseez/opencv/blob/2f4e38c8313ff313de7c41141d56d945d91f47cf/modules/imgproc/src/drawing.cpp#L1411
    # See test_circle_drawing
    # (---+---) or (---+--)
    w = r*2
    h = w
    return (x,y,w,h)

def extract_circle(img, circle, margin=0, mask_color=None):
    (cx, cy), r = circle
    x, y, h, w = circle_bb(((cx, cy), r+margin))
    roi = img[y:y+h, x:x+w]
    if mask_color:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx-x, cy-y), r, 255, -1)
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

    # ball_bbs = ([two_point_rect_to_bb(*bb) for bb in metadata["ball_bbs"]])
    # ball_bbs = sorted(ball_bbs, key=lambda bb: bb[2:])
    # ball_bbs = np.array(ball_bbs)
    balls = np.array(sorted(metadata["ball_circles"], key=lambda c: c[1]))

    ball_count = len(balls)

    border = 2
    cols = ball_count
    rows = math.ceil(ball_count / cols)

    req_ball_w = balls[:,1].sum()*2 # inexact for multi-row, but safe and easy
    req_ball_h = balls[:,1].max()*2 # inexact for multi-row, but safe and easy

    canvas_shape = ((req_ball_h+border)*rows, req_ball_w+ball_count*border, 3)
    canvas = np.zeros(canvas_shape,
                      dtype=np.float32)

    scale = 1
    target_size = (canvas.shape[1]*scale, canvas.shape[0]*scale)

    frame = np.zeros((1024, 1024, 3), dtype=np.float32)

    timer = utilities.Timer()
    for i, file_path in enumerate(filenames):
        img = Image(file_path, color_normalization=False)

        pix = img.bgr

        tx, ty = 0, 0
        col = 0
        for j, circle in enumerate(balls):
            (x,y), r = circle
            h = r*2
            w = h
            # One would think that creating the circle mask repeatedly etc. is slow
            # but the image loading is way to dominant to care about that
            canvas[ty:ty+h, tx:tx+w] = extract_circle(pix, circle, mask_color=(0,0,0))

            # x,y,w,h = circle_bb(circle)
            # canvas[ty:ty+h, tx:tx+w] = pix[y:y+h, x:x+w]

            tx += w+border
            col += 1
            if col == cols:
                col = 0
                tx = 0
                ty += req_ball_h + border

        resized = canvas if scale == 1 else cv2.resize(canvas, target_size)
        ox, oy = 0, 100
        frame[oy:oy+resized.shape[0], ox:ox+resized.shape[1]] = resized

        ## Draw "ghost" histograms
        x = ox
        y = oy+resized.shape[0]+10

        for ch in range(3):
            hist_img = utilities.draw_histogram(canvas[:,:,ch],
                                                ignored_values=[0],
                                                fg=255/len(filenames)*1.5, bg=0)

            # dedicate one color channel to each third of the timelapse:
            k = i // (math.ceil(len(filenames) / 3))
            h, w = hist_img.shape
            frame[y:y+h, x:x+w, k] += hist_img
            # frame[y:y+h, x:x+w, 1] = frame[y:y+h, x:x+w, 0]
            # frame[y:y+h, x:x+w, 2] = frame[y:y+h, x:x+w, 0]

            x += hist_img.shape[1]+5

        if i%5 == 0:
            utilities.show(frame, time_ms=10, text=os.path.basename(file_path))
            print(timer)
            timer.reset()

    while True:
        # q to exit
        utilities.show(frame)


