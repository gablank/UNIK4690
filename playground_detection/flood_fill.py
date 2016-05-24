#!/usr/bin/python3
import cv2
import numpy as np
import utilities
import transformer
from image import Image


class FloodFillPlaygroundDetector:
    def __init__(self, petanque_detection):
        self.petanque_detection = petanque_detection
        self.transformer = transformer.Transformer(filename="playground_transformer_state.json")

    def detect(self, image):
        best_transformation = self.transformer.get_playground_transformation(image)
        # cv2.imwrite("/home/anders/UNIK4690/project/report/playground_detection_pipeline/step_2_best_transformation.png", utilities.as_uint8(best_transformation))
        # return

        best_transformation = utilities.as_uint8(best_transformation)
        middle = utilities.get_middle(best_transformation)
        box = utilities.get_box(best_transformation, middle, 100)
        if np.average(box) > np.average(best_transformation):
            print("Inverting image")
            best_transformation = 255 - best_transformation
            box = utilities.get_box(best_transformation, middle, 100)

        # cv2.imwrite("/home/anders/UNIK4690/project/report/playground_detection_pipeline/step_3_invert.png", utilities.as_uint8(best_transformation))
        blur_size = 5
        best_transformation = cv2.blur(best_transformation, (blur_size, blur_size))
        # cv2.imwrite("/home/anders/UNIK4690/project/report/playground_detection_pipeline/step_4_blur.png", utilities.as_uint8(best_transformation))

        box_hist = utilities.get_histogram(box)
        box_hist /= sum(box_hist)
        argmax = np.argmax(box_hist)
        cur = argmax
        low = argmax
        high = argmax
        tot_sum = 0.0
        while tot_sum < 0.9:
            tot_sum += box_hist[cur]
            if high == 255:
                cur = low - 1
            elif low == 0:
                cur = high + 1
            else:
                if box_hist[high+1] > box_hist[low-1]:
                    cur = high + 1
                else:
                    cur = low - 1
            low = min(low, cur)
            high = max(high, cur)

        def threshold_range(img, lo, hi):
            th_lo = cv2.threshold(img, lo, 255, cv2.THRESH_BINARY)[1]
            th_hi = cv2.threshold(img, hi, 255, cv2.THRESH_BINARY_INV)[1]
            return cv2.bitwise_and(th_lo, th_hi)
        print("Thresholding between {} and {}".format(low, high))

        best_transformation = threshold_range(best_transformation, low, high)
        # cv2.imwrite("/home/anders/UNIK4690/project/report/playground_detection_pipeline/step_5_threshold.png", utilities.as_uint8(best_transformation))

        # ret, best_transformation = cv2.threshold(best_transformation, idx, 255, cv2.THRESH_TOZERO)
        box = utilities.get_box(best_transformation, middle, 100)
        box[:,:] = 255

        best_transformation[np.where(best_transformation == 255)] = 254

        num_filled, best_transformation, _, _ = cv2.floodFill(best_transformation, None, middle, 255, upDiff=0, loDiff=0, flags=cv2.FLOODFILL_FIXED_RANGE)

        best_transformation[np.where(best_transformation != 255)] = 0
        # cv2.imwrite("/home/anders/UNIK4690/project/report/playground_detection_pipeline/step_6_flood_fill.png", utilities.as_uint8(best_transformation))

        kernel_size = 3
        iterations = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        best_transformation = cv2.erode(best_transformation, kernel, iterations=iterations)
        best_transformation = cv2.dilate(best_transformation, kernel, iterations=iterations-1)
        # cv2.imwrite("/home/anders/UNIK4690/project/report/playground_detection_pipeline/step_7_opening.png", utilities.as_uint8(best_transformation))

        best_transformation[np.where(best_transformation == 255)] = 254
        num_filled, best_transformation, _, _ = cv2.floodFill(best_transformation, None, middle, 255, upDiff=0, loDiff=0, flags=cv2.FLOODFILL_FIXED_RANGE)
        best_transformation[np.where(best_transformation != 255)] = 0
        # cv2.imwrite("/home/anders/UNIK4690/project/report/playground_detection_pipeline/step_8_flood_fill_2.png", utilities.as_uint8(best_transformation))

        im2, contours, hierarchy = cv2.findContours(best_transformation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # IDEA: For each vertex, see if the density inside the triangle made by connecting this vertex with the previous
            # and next vertices are much less than the density of the entire convex hull.
            # If it is, we should probably remove this vertex as it is the result
            #  of a "thin arm" that the flood fill followed.
            polygon = contours[0]
            for idx in range(1, len(contours)):
                polygon = np.concatenate((polygon, contours[idx]))

            convex_hull = cv2.convexHull(polygon)

            convex_hull = cv2.approxPolyDP(convex_hull, 5, True)
            original_convex_hull = convex_hull.copy()

            convex_hull_as_list = []
            for idx in range(len(convex_hull)):
                vertex = convex_hull[idx][0]
                convex_hull_as_list.append((vertex[0], vertex[1]))

            temp = best_transformation.copy()
            temp[np.where(temp == 255)] = 120
            for idx, v1 in enumerate(convex_hull_as_list):
                cv2.line(temp, v1, convex_hull_as_list[(idx+1)%len(convex_hull_as_list)], 255, 3)
            # cv2.imwrite("/home/anders/UNIK4690/project/report/playground_detection_pipeline/step_9_convex_hull.png", utilities.as_uint8(temp))

            convex_hull_mask = utilities.poly2mask(convex_hull, best_transformation)
            best_transformation[np.where(convex_hull_mask == 0)] = 0
            convex_hull_n = np.count_nonzero(convex_hull_mask)
            convex_hull_n_white = np.count_nonzero(best_transformation[np.where(convex_hull_mask != 0)])
            convex_hull_density = convex_hull_n_white / convex_hull_n
            # print("Convex hull density:", convex_hull_density)
            copy = best_transformation.copy()

            # utilities.show(convex_hull_mask, time_ms=10)
            outside_convex_hull_color = 0
            outside_triangle_not_filled_color = int(1/6 * 255)
            outside_triangle_filled_color = int(2/3 * 255)
            inside_triangle_not_filled_color = int(2/6 * 255)
            inside_triangle_filled_color = 255

            copy[np.where(convex_hull_mask == 0)] = outside_convex_hull_color
            copy[np.where((convex_hull_mask != 0) & (copy != 0))] = outside_triangle_filled_color
            copy[np.where((convex_hull_mask != 0) & (copy == 0))] = outside_triangle_not_filled_color

            # utilities.show(best_transformation, time_ms=0)

            new_convex_hull = []

            #import transformer
            #from image import Image
            #light_mask = transformer.Transformer.get_light_mask(Image(image_data=img))

            idx = 0
            isd = 0
            while idx < len(convex_hull_as_list):
                temp = best_transformation.copy()
                temp[np.where(temp == 255)] = 120

                v1 = convex_hull_as_list[(idx-1)%len(convex_hull_as_list)]
                v2 = convex_hull_as_list[idx]
                v3 = convex_hull_as_list[(idx+1)%len(convex_hull_as_list)]

                triangle_mask = utilities.poly2mask([v1, v2, v3], best_transformation)

                # temp[np.where((triangle_mask != 0) & (temp != 0))] = 255
                # temp[np.where((triangle_mask != 0) & (temp == 0))] = 60
                # utilities.show(temp)
                # cv2.imwrite("/home/anders/UNIK4690/project/report/playground_detection_pipeline/step_10_{}_convex_hull_refinement.png".format(isd), utilities.as_uint8(temp))
                isd += 1
                copy[np.where((triangle_mask == 0) & (copy == inside_triangle_filled_color))] = outside_triangle_filled_color
                copy[np.where((triangle_mask == 0) & (copy == inside_triangle_not_filled_color))] = outside_triangle_not_filled_color

                copy[np.where((triangle_mask != 0) & (copy == outside_triangle_filled_color))] = inside_triangle_filled_color
                copy[np.where((triangle_mask != 0) & (copy == outside_triangle_not_filled_color))] = inside_triangle_not_filled_color

                triangle_n_white = np.count_nonzero(best_transformation[np.where(triangle_mask != 0)])
                triangle_n = np.count_nonzero(triangle_mask)
                triangle_density = triangle_n_white / triangle_n

                #light_density = np.count_nonzero(light_mask[np.where(triangle_mask != 0)]) / triangle_n

                limit = 0.4*convex_hull_density# + light_density / 2
                # utilities.show(copy, text="Triangle density: {}, limit: {}".format(round(triangle_density, 2), round(limit, 2)))
                if triangle_density > limit:
                    new_convex_hull.append(v2)
                    idx += 1
                else:
                    best_transformation[np.where(triangle_mask != 0)] = 0
                    convex_hull_as_list.remove(v2)
                    copy[np.where(triangle_mask != 0)] = outside_convex_hull_color

            def show_lines(image, pts):
                # to_show = image.get_bgr().copy()
                to_show = utilities.as_float32(image.original_bgr)
                # to_show = image.copy()
                # to_show[:,:] = 0
                for idx, pt1 in enumerate(pts):
                    if idx % 2 == 0:
                        color = (0, 0, 1)
                    else:
                        color = (1, 0, 0)
                    pt2 = pts[(idx+1) % len(pts)]
                    cv2.line(to_show, pt1, pt2, color, 3)
                # cv2.imwrite("/home/anders/UNIK4690/project/report/playground_detection_pipeline/step_11_result_2.png", utilities.as_uint8(to_show))
                utilities.show(to_show, text=image.filename, time_ms=30)

            # show_lines(best_transformation, new_convex_hull)

            convex_hull = []
            for idx, pt in enumerate(new_convex_hull):
                angle = utilities.get_angle(new_convex_hull[(idx-1)%len(new_convex_hull)], pt, new_convex_hull[(idx+1)%len(new_convex_hull)])
                if 15 < angle < 180-15:
                    convex_hull.append(pt)

            new_convex_hull = convex_hull
            # show_lines(image, new_convex_hull)

            angles = []
            for idx, pt in enumerate(new_convex_hull):
                angles.append((pt, utilities.get_angle(new_convex_hull[(idx-1)%len(new_convex_hull)], pt, new_convex_hull[(idx+1)%len(new_convex_hull)])))

            angles.sort(key=lambda x: x[1], reverse=True)

            lines = []
            for idx, pt in enumerate(new_convex_hull):
                pt2 = new_convex_hull[(idx+1)%len(new_convex_hull)]
                lines.append((pt, pt2, idx))

            # sort by line length
            lines.sort(key=lambda x: utilities.distance(x[0], x[1]))

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

            print(points)

            lines = []
            for idx, pt1 in enumerate(points):
                lines.append((pt1, points[(idx + 1) % len(points)]))

            # sort by line length
            lines.sort(key=lambda x: (x[0][0]-x[1][0])**2 + (x[0][1]-x[1][1])**2)

            print("Lines:", lines)

            # shortest line is probably one of the short sides
            short_side = lines[0]
            # sides_in_order: First one of the short sides, then a long side, then the other short side, then the last long side
            sides_in_order = [short_side[0], short_side[1]]
            while len(sides_in_order) < 4:
                for line in lines:
                    if len(sides_in_order) >= 4:
                        break
                    if sides_in_order[-1] == line[0]:
                        sides_in_order.append(line[1])
            print(sides_in_order)
            show_lines(image, sides_in_order)
            return sides_in_order

        return


if __name__ == "__main__":
    flood_fill = FloodFillPlaygroundDetector(None)

    # try:
    import os
    filenames = []
    for cur in os.walk(os.path.join(utilities.get_project_directory(), "images/raspberry/south/")):
        filenames = cur[2]
        break

    filenames.sort()

    try:
        for file in filenames:
            try:
                import datetime
                date = datetime.datetime.strptime(file, "%Y-%m-%d_%H:%M:%S.png")
                # if date < datetime.datetime(2016, 4, 13, 7, 5):
                # if date < datetime.datetime(2016, 4, 12, 19, 0):
                #     continue
                image = Image(file, histogram_equalization="rg_chromaticity")
                # image = Image(file, histogram_equalization=None)
                # utilities.show_all(image, time_ms=30)
                # cv2.imwrite("/home/anders/UNIK4690/project/report/playground_detection_pipeline/step_0_original.png", Image(file, histogram_equalization=None).get_bgr(np.uint8))
                # cv2.imwrite("/home/anders/UNIK4690/project/report/playground_detection_pipeline/step_1_rg_chromaticity.png", image.get_bgr(np.uint8))
                # utilities.show_all(image, time_ms=30)
                # continue
            except FileNotFoundError:
                continue
            except ValueError:
                continue

            flood_fill.detect(image)
            # exit(0)

    except Exception as e:
        import traceback
        print(e)
        traceback.print_tb(e.__traceback__)

    finally:
        flood_fill.transformer.save(filename="playground_transformer_state.json")
