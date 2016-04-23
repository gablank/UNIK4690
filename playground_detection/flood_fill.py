#!/usr/bin/python3
import cv2
import numpy as np
import utilities
import transformer


class FloodFillPlaygroundDetector:
    def __init__(self, petanque_detection):
        self.petanque_detection = petanque_detection
        self.transformer = transformer.Transformer(filename="playground_transformer_state.json")

    def detect(self, image):
        best_transformation = self.transformer.get_playground_transformation(image)

        best_transformation = utilities.as_uint8(best_transformation)
        middle = utilities.get_middle(best_transformation)
        box = utilities.get_box(best_transformation, middle, 100)
        if np.average(box) > np.average(best_transformation):
            print("Inverting image")
            best_transformation = 255 - best_transformation

        blur_size = 5
        best_transformation = cv2.blur(best_transformation, (blur_size, blur_size))

        hist = cv2.calcHist([best_transformation], [0], None, [256], [0, 256])
        hist /= sum(hist)

        tot = 0.0
        idx = 0

        while tot < 0.3:
            tot += hist[idx]
            idx += 1

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
            low = min(cur, low)
            high = max(cur, high)

        def threshold_range(img, lo, hi):
            th_lo = cv2.threshold(img, lo, 255, cv2.THRESH_BINARY)[1]
            th_hi = cv2.threshold(img, hi, 255, cv2.THRESH_BINARY_INV)[1]
            return cv2.bitwise_and(th_lo, th_hi)
        print("Thresholding between {} and {}".format(low, high))
        best_transformation = threshold_range(best_transformation, low, high)
        # ret, best_transformation = cv2.threshold(best_transformation, idx, 255, cv2.THRESH_TOZERO)
        box = utilities.get_box(best_transformation, middle, 100)
        box[:,:] = 255
        # utilities.show(best_transformation)

        best_transformation[np.where(best_transformation == 255)] = 254

        # utilities.show(best_transformation, "transformed image", fullscreen=True, time_ms=0)

        num_filled, best_transformation, _, _ = cv2.floodFill(best_transformation, None, middle, 255, upDiff=0, loDiff=0, flags=cv2.FLOODFILL_FIXED_RANGE)

        best_transformation[np.where(best_transformation != 255)] = 0
        # utilities.show(best_transformation)

        # utilities.show(best_transformation)
        # utilities.show(img)

        kernel_size = 3
        iterations = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        best_transformation = cv2.erode(best_transformation, kernel, iterations=iterations)
        best_transformation = cv2.dilate(best_transformation, kernel, iterations=iterations-1)

        best_transformation[np.where(best_transformation == 255)] = 254
        num_filled, best_transformation, _, _ = cv2.floodFill(best_transformation, None, middle, 255, upDiff=0, loDiff=0, flags=cv2.FLOODFILL_FIXED_RANGE)
        best_transformation[np.where(best_transformation != 255)] = 0

        # utilities.show(transformed, time_ms=10)
        #utilities.show(transformed, "transformed image", fullscreen=True, time_ms=10)

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
            # temp = utilities.draw_convex_hull(img, convex_hull)
            # utilities.show(temp, "contours")

            convex_hull = cv2.approxPolyDP(convex_hull, 5, True)
            # temp = utilities.draw_convex_hull(img, convex_hull)
            # utilities.show(temp, "contours")

            convex_hull_as_list = []
            for idx in range(len(convex_hull)):
                vertex = convex_hull[idx][0]
                convex_hull_as_list.append((vertex[0], vertex[1]))


            convex_hull_mask = utilities.poly2mask(convex_hull, best_transformation)
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
            while idx < len(convex_hull_as_list):
                v1 = convex_hull_as_list[(idx-1)%len(convex_hull_as_list)]
                v2 = convex_hull_as_list[idx]
                v3 = convex_hull_as_list[(idx+1)%len(convex_hull_as_list)]

                triangle_mask = utilities.poly2mask([v1, v2, v3], best_transformation)

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
                    convex_hull_as_list.remove(v2)
                    copy[np.where(triangle_mask != 0)] = outside_convex_hull_color

            # for i in range(len(convex_hull)):
            #     angle = utilities.get_angle(convex_hull[(i-1)%len(convex_hull)][0], convex_hull[i][0], convex_hull[(i+1)%len(convex_hull)][0])
            #     if 25 < angle < 170:
            #         new_convex_hull.append(convex_hull[i])
            # convex_hull = np.array(new_convex_hull)

            return new_convex_hull

        return
