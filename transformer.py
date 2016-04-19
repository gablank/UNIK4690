#!/usr/bin/python3
import numpy as np
import utilities
import cv2
from image import Image


def discriminatory_power(transform):
    box = utilities.get_box(transform, utilities.get_middle(transform), 100)
    box_mean, box_std_dev = cv2.meanStdDev(box)
    all_mean, all_std_dev = cv2.meanStdDev(transform)
    # Return format is a bit weird ...
    box_var = box_std_dev[0][0]**2
    all_var = all_std_dev[0][0]**2
    box_mean = box_mean[0][0]
    all_mean = all_mean[0][0]
    return (box_mean - all_mean)**2 / box_var


def mean_diff(transform):
    box = utilities.get_box(transform, utilities.get_middle(transform), 100)
    box_mean, box_std_dev = cv2.meanStdDev(box)
    all_mean, all_std_dev = cv2.meanStdDev(transform)
    # print(abs(box_mean - all_mean), 70*box_std_dev**2)
    return abs(box_mean - all_mean) - 70*box_std_dev**2


def ball_fitness_func(transform):
    return 1.0


class Transformation:
    def __init__(self, coefficients, fitness_func):
        self.coefficients = coefficients
        self.fitness_func = fitness_func

    def fitness(self, image):
        transform = self.get_transform(image)
        return self.fitness_func(transform)

    def get_transform(self, image):
        color_spaces = (image.get_bgr(), image.get_hsv(), image.get_lab(), image.get_ycrcb())
        return utilities.transform_image(color_spaces, self.coefficients)


class Transformer:
    def __init__(self, initial_playground_transformation_coefficients, initial_ball_transformations_coefficients):
        self.playground_fitness_func = mean_diff
        self.playground_transformations = []
        for c in initial_playground_transformation_coefficients:
            self.playground_transformations.append(Transformation(c, self.playground_fitness_func))

        self.ball_transformations = []
        for c in initial_ball_transformations_coefficients:
            self.ball_transformations.append(Transformation(c, ball_fitness_func))

    def get_playground_transformation(self, image):
        """
        Get the optimal transform for finding the playground
        """

        # Sort transformations according to their fitness for this image
        self.playground_transformations.sort(key=lambda transform: transform.fitness(image))

        cur_best_transform = self.playground_transformations[-1]
        cur_best_fitness = cur_best_transform.fitness(image)

        # Perform a quick hill climber before returning
        step_size = 0.1
        max_iters = 1
        for _ in range(max_iters):
                for i in range(len(cur_best_transform.coefficients)):
                    c1 = cur_best_transform.coefficients.copy()
                    c2 = cur_best_transform.coefficients.copy()

                    # If c1[i] is 0.0 we won't ever change the value unless we add a minimum value like here
                    c1[i] += max(0.05, step_size * c1[i])
                    c2[i] -= max(0.05, step_size * c2[i])

                    t1 = Transformation(c1, self.playground_fitness_func)
                    t2 = Transformation(c2, self.playground_fitness_func)

                    t1_fitness = t1.fitness(image)
                    t2_fitness = t2.fitness(image)

                    if t1_fitness > cur_best_fitness:
                        cur_best_transform = t1
                        cur_best_fitness = t1_fitness
                        # break

                    if t2_fitness > cur_best_fitness:
                        cur_best_transform = t2
                        cur_best_fitness = t2_fitness
                        # break

        # Update the previous best
        print("Updating transform: {} to {}".format(self.playground_transformations[-1].coefficients, cur_best_transform.coefficients))
        self.playground_transformations[-1] = cur_best_transform
        return cur_best_transform.get_transform(image)


if __name__ == "__main__":
    # This one seems VERY good: [0.06088760671276526, -1.3561395049862874, 0.09000000000000001, -2.0728250555077863, 1.1970045975476478, -0.013731644143073622, 0.01089875092195508, 0.011162617086235636, -0.0779020312300068, -1.1679306731502184, 1.1968644184452055]
    initial_playground_transformation_coefficients = (
        [0.055352369738877506, -1.506821672206986, 0.1, -1.8843864140979874, 1.0881859977705888, -0.015257382381192912, 0.009907955383595526, 0.01240290787359515, -0.08655781247778534, -1.2977007479446871, 1.088058562222914],
        [-0.1988585218329525, -4.5146823917495364, 0.01090267776428841, 1.313290643396085, 1.0563778245986046, 0.6002346526970821, -0.25974518055190304, -0.933624230536284, 0.6822183925261562, 0.7653565499049495, 2.1643479659811984],
        [-1.111020254026619, -2.7742975576709337, 0.000270181868861975, 0.02662770269975154, 0.026178351873268172, 0.0606031669097949, -8.832361764989198, -0.01892975372132204, 0.08418754679794245, 17.421791129064555, 0.04388336615871081],
        [-0.9010202540266188, -2.48429755767094, -2.559455355134354, -0.5933722973002487, 0.43397727125388325, 1.8314429990602692, -9.932361764989174, 2.32831130753861, 0.20716529330623937, 13.598730744280122, -0.8061166338412897]
    )
    initial_ball_transformation_coefficients = (
    )

    transformer = Transformer(initial_playground_transformation_coefficients, initial_ball_transformation_coefficients)

    import os
    filenames = []
    for cur in os.walk(os.path.join(utilities.get_project_directory(), "images/microsoft_cam/24h/south/")):
        filenames = cur[2]
        break

    filenames.sort()

    for file in filenames:
        try:
            img = Image(os.path.join(utilities.get_project_directory(), "images/microsoft_cam/24h/south/", file))
        except FileNotFoundError:
            continue

        best_transform = transformer.get_playground_transformation(img)
        import playground_detection
        playing_field = playground_detection.detect(img.get_bgr(np.uint8), best_transform, "flood_fill", draw_field=True)

        # utilities.show(best_transform, time_ms=10, text=img.filename)
