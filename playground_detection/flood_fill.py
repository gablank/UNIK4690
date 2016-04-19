#!/usr/bin/python3
import cv2
import numpy as np
import utilities


def detect(img):
    # if img.dtype != np.float32:
    #     img = img.astype(np.float32)
    #     img /= 255
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # transform_vector = [0.3756748599999996, 1.3100668899999999, 3.65993528, 0.000728242339, -12.494303939999975, -1.41144737, -0.484814647, 0.125014553, 0.696660682, 0.332596334, -5.443031946999997, -4.181805536000002]
    # transformed = utilities.transform_image((img, hsv, lab, ycrcb), transform_vector)

    #utilities.show(transformed, "transformed image", fullscreen=True)

    transformed = utilities.as_uint8(img)
    middle = utilities.get_middle(transformed)
    box = utilities.get_box(transformed, middle, 100)

    blur_size = 5
    transformed = cv2.blur(transformed, (blur_size, blur_size))
    hist = cv2.calcHist([transformed], [0], None, [256], [0, 256])
    hist /= sum(hist)

    tot = 0.0
    i = 0
    increments = 1
    if np.average(box) > np.average(transformed):
        i = 255
        increments = -1
    while tot < 0.3:
        tot += hist[i]
        i += increments

    ret, transformed = cv2.threshold(transformed, i, 255, cv2.THRESH_TOZERO)
    box[:,:] = 0
    utilities.show(transformed, time_ms=10)

    transformed[np.where(transformed == 255)] = 254


    #utilities.show(transformed, "transformed image", fullscreen=True, time_ms=10)

    num_filled, transformed, _, _ = cv2.floodFill(transformed, None, middle, 255, upDiff=0, loDiff=0, flags=cv2.FLOODFILL_FIXED_RANGE)

    transformed[np.where(transformed != 255)] = 0

    # kernel_size = 7
    # iterations = 3
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    # transformed = cv2.erode(transformed, kernel, iterations=iterations)
    # transformed = cv2.dilate(transformed, kernel, iterations=iterations-1)
    #
    # utilities.show(transformed, time_ms=10)
    #utilities.show(transformed, "transformed image", fullscreen=True, time_ms=10)

    im2, contours, hierarchy = cv2.findContours(transformed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # IDEA: For each vertex, see if the density inside the triangle made by connecting this vertex with the previous
        # and next vertices are much less than the density of the entire convex hull.
        # If it is, we should probably remove this vertex as it is the result
        #  of a "thin arm" that the flood fill followed.
        polygon = contours[0]
        for i in range(1, len(contours)):
            polygon = np.concatenate((polygon, contours[i]))

        convex_hull = cv2.convexHull(polygon)
        # temp = utilities.draw_convex_hull(img, convex_hull)
        # utilities.show(temp, "contours")

        convex_hull = cv2.approxPolyDP(convex_hull, 5, True)
        # temp = utilities.draw_convex_hull(img, convex_hull)
        # utilities.show(temp, "contours")

        convex_hull_as_list = []
        for i in range(len(convex_hull)):
            vertex = convex_hull[i][0]
            convex_hull_as_list.append((vertex[0], vertex[1]))


        convex_hull_mask = utilities.poly2mask(convex_hull, transformed)
        convex_hull_n = np.count_nonzero(convex_hull_mask)

        # utilities.show(convex_hull_mask, time_ms=10)
        transformed[np.where(convex_hull_mask == 0)] = 120

        convex_hull_n_white = np.count_nonzero(transformed[np.where(convex_hull_mask != 0)])
        convex_hull_density = convex_hull_n_white / convex_hull_n
        print("Convex hull density:", convex_hull_density)

        utilities.show(transformed, time_ms=0)

        new_convex_hull = []
        for i in range(len(convex_hull)):
            v1 = convex_hull[(i-1)%len(convex_hull)]
            v2 = convex_hull[i]
            v3 = convex_hull[(i+1)%len(convex_hull)]

            triangle_mask = utilities.poly2mask([v1, v2, v3], transformed)
            triangle_n_white = np.count_nonzero(transformed[np.where(triangle_mask != 0)])
            triangle_n = np.count_nonzero(triangle_mask)
            triangle_density = triangle_n_white / triangle_n
            print(triangle_density)
            if triangle_density > 0.3*convex_hull_density:
                print("Density above 0.5, adding")
                new_convex_hull.append(v2)
        convex_hull = np.array(new_convex_hull)


        # for i in range(len(convex_hull)):
        #     angle = utilities.get_angle(convex_hull[(i-1)%len(convex_hull)][0], convex_hull[i][0], convex_hull[(i+1)%len(convex_hull)][0])
        #     if 25 < angle < 170:
        #         new_convex_hull.append(convex_hull[i])
        # convex_hull = np.array(new_convex_hull)

        return convex_hull

    return

    box = utilities.get_box(transformed, middle, 40)
    amin = np.amin(box)
    amax = np.amax(box)
    #ret, transformed = cv2.threshold(transformed, 100, 255, cv2.THRESH_BINARY)
    #from matplotlib import pyplot as plt
    #utilities.plot_histogram(box)
    #utilities.plot_histogram(transformed)
    #plt.show()
    #return
    ret, transformed = cv2.threshold(transformed, 50, 255, cv2.THRESH_BINARY)
    #transformed = 255 - transformed
    #ret, transformed = cv2.threshold(transformed, 255-amin, 255, cv2.THRESH_TOZERO)
    utilities.show(transformed, "transformed image", fullscreen=True, time_ms=10)
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)

    #hue = hsv[:,:,0] / np.amax(hsv[:,:,0])

    transformed = transformed.astype(np.float32)
    transformed /= 255
    #utilities.show(transformed, "transformed image", fullscreen=True, time_ms=10)
    #return

    hue = transformed
    #hue = cv2.blur(hue, (3,3))
    #hue = hue**16

    hue = hue * 255
    hue = hue.astype(np.uint8)

    #utilities._show(hue)
    iterations = 2
    size = 15

    dilationElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    hue = cv2.dilate(hue, dilationElement, iterations=iterations)

    utilities.show(hue, "after dilation", fullscreen=True)
    hue[np.where(hue == 255)] = 254
    box_size = 30
    hue[middle[1]-box_size:middle[1]+box_size, middle[0]-box_size:middle[0]+box_size] = 0
    mask = utilities.flood_fill_until(hue, 0.2)
    # diff = 2
    # num_filled = 0
    # while num_filled < 0.2*(len(img) * len(img[0])):
    #     num_filled, mask, _, _ = cv2.floodFill(hue.copy(), None, middle, 255, upDiff=diff, loDiff=diff, flags=cv2.FLOODFILL_FIXED_RANGE)
    #     diff += 1
    # utilities.show(mask, "mask")

    # hue[np.where(mask != 255)] = 0
    # utilities.show(hue, "hue")
    # return

    # hue *= 255
    # hue = hue.astype(np.uint8)
    # ret, hue = cv2.threshold(hue, 1, 255, cv2.THRESH_BINARY)
    #
    # utilities.show(hue)

    # im2, contours, hierarchy = cv2.findContours(hue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        polygon = contours[0]
        for i in range(1, len(contours)):
            polygon = np.concatenate((polygon, contours[i]))

        convex_hull = cv2.convexHull(polygon)
        # temp = utilities.draw_convex_hull(img, convex_hull)
        # utilities.show(temp, "contours")

        convex_hull = cv2.approxPolyDP(convex_hull, 5, True)
        # temp = utilities.draw_convex_hull(img, convex_hull)
        # utilities.show(temp, "contours")

        new_convex_hull = []
        for i in range(len(convex_hull)):
            if utilities.get_angle(convex_hull[(i-1)%len(convex_hull)][0], convex_hull[i][0], convex_hull[(i+1)%len(convex_hull)][0]) > 25:
                new_convex_hull.append(convex_hull[i])
        convex_hull = np.array(new_convex_hull)

        return convex_hull
