#!/usr/bin/python3
import cv2
import numpy as np
import playground_detection.maximize_density as maximize_density
import playground_detection.flood_fill as flood_fill


def detect(img, method="flood_fill", draw_field=False):
    polygon = [(0, 0), (len(img)-1, 0), (len(img)-1, len(img[0])-1), (0, len(img[0])-1)]
    if method == "flood_fill":
        polygon = flood_fill.detect(img)
    elif method == "maximize_density":
        polygon = maximize_density.detect(img)

    if polygon is None:
        return img

    g_b = img[:,:,0].copy()
    g_b[:,:] = 0
    g_b = cv2.fillConvexPoly(g_b, polygon, color=255)

    playing_field = img.copy()

    lines = []
    for idx in range(len(polygon)):
        if method == "maximize_density":
            pt1 = (polygon[idx][0], polygon[idx][1])
            pt2 = (polygon[(idx+1)%len(polygon)][0], polygon[(idx+1)%len(polygon)][1])
        else:
            pt1 = polygon[idx][0]
            pt2 = polygon[(idx+1)%len(polygon)][0]
            pt1 = (pt1[0], pt1[1])
            pt2 = (pt2[0], pt2[1])
        lines.append((pt1, pt2, idx))

    # sort by line length
    lines.sort(key=lambda x: (x[0][0]-x[1][0])**2 + (x[0][1]-x[1][1])**2)

    top_four = list(lines[-4:])
    top_four.sort(key=lambda x: x[2])

    points = []
    for idx, line in enumerate(top_four):
        next = top_four[(idx+1) % len(top_four)]
        l1_p1, l1_p2, line_idx = line[0], line[1], line[2]
        l2_p1, l2_p2, next_idx = next[0], next[1], next[2]

        #print("From", l1_p1, "to", l1_p2)
        #print("From", l2_p1, "to", l2_p2)
        #print()

        # cv2.circle(img, l1_p1, 9, (0,0,255), 5)
        # cv2.circle(img, l1_p2, 9, (0,255,0), 5)

        l1_p1x, l1_p2x = l1_p1[0], l1_p2[0]
        l2_p1x, l2_p2x = l2_p1[0], l2_p2[0]

        l1_p1y, l1_p2y = l1_p1[1], l1_p2[1]
        l2_p1y, l2_p2y = l2_p1[1], l2_p2[1]

        def intersection(x1, y1, x2, y2, x3, y3, x4, y4):
            Px_upper = np.array([x1, y1, x1, 1, x2, y2, x2, 1, x3, y3, x3, 1, x4, y4, x4, 1]).reshape((4,4))
            lower = np.array([x1, 1, y1, 1, x2, 1, y2, 1, x3, 1, y3, 1, x4, 1, y4, 1]).reshape((4,4))

            Py_upper = Px_upper.copy()
            Py_upper[:,2] = Py_upper[:,1].copy()

            _det = np.linalg.det
            def det(mat):
                return _det(mat[:2, :2]) * _det(mat[2:, 2:]) - _det(mat[2:, :2]) * _det(mat[:2, 2:])

            return (int(round(det(Px_upper) / det(lower), 0)), int(round(det(Py_upper) / det(lower), 0)))


        point = intersection(l1_p1x, l1_p1y, l1_p2x, l1_p2y, l2_p1x, l2_p1y, l2_p2x, l2_p2y)
        points.append(point)

    lines = []
    for idx, pt1 in enumerate(points):
        lines.append((pt1, points[(idx + 1) % len(points)]))

    # sort by line length
    lines.sort(key=lambda x: (x[0][0]-x[1][0])**2 + (x[0][1]-x[1][1])**2)

    # shortest line is one of the short sides
    short_side = lines[0]
    # sides_in_order: First one of the short sides, then a long side, then the other short side, then the last long side
    sides_in_order = [short_side]
    while len(sides_in_order) < 4:
        for line in lines:
            if line in sides_in_order:
                continue
            # print(sides_in_order[-1][1], line[0])
            if sides_in_order[-1][1] == line[0] or sides_in_order[-1][1] == line[1]:
                sides_in_order.append(line)

    # Origo is defined as the "clockwise-most point in the longest short side (so sides_in_order[2])
    _, origo = sides_in_order[2]

    draw_origo = True
    if draw_origo:
        cv2.circle(img, origo, 9, (0,255,0), 3)

    for idx, line in enumerate(sides_in_order):
        if idx % 2 == 0:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        pt1, pt2 = line
        cv2.line(img, pt1, pt2, color, 3)

    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.imshow("test", img)
    # key = cv2.waitKey(0)
    # if key == 113:
    #     exit(0)

    # Use g_b as a "mask" for img (I couldn't figure out the proper way to use it as a mask)
    for i in (0, 1, 2):
        playing_field[:,:,i] = np.minimum(g_b, playing_field[:,:,i])

    playground_width = 2.5
    playground_length = 5.0
    scale = int(min(1080/playground_width, 1920/playground_length))
    playground_width_scaled = playground_width * scale
    playground_length_scaled = playground_length * scale
    real_world_coordinates = [(0, 0), (playground_length_scaled, 0), (playground_length_scaled, playground_width_scaled), (0, playground_width_scaled)]
    points_in_order = []
    for side in sides_in_order[3:] + sides_in_order[:3]:
        points_in_order.append(side[0])

    M, mask = cv2.findHomography(np.array(points_in_order), np.array(real_world_coordinates), method=cv2.RANSAC)

    matches_mask = mask.ravel().tolist()
    if matches_mask == [1, 1, 1, 1]:

        playing_field = cv2.warpPerspective(playing_field, M, (1920, 1080))

        # cv2.imshow("test", playing_field)
        # key = cv2.waitKey(30)
        # if key == 113:
        #     exit(0)


    #cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.imshow("test", playing_field)
    #cv2.waitKey(0)
    return playing_field
