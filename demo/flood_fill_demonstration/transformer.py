#!/usr/bin/python3
import numpy as np
import utilities
import cv2
from image import Image
import json
import copy


def discriminatory_power(grayscale):
    box = utilities.get_box(grayscale, utilities.get_middle(grayscale), 100)
    box_mean, box_std_dev = cv2.meanStdDev(box)
    all_mean, all_std_dev = cv2.meanStdDev(grayscale)
    # Return format is a bit weird ...
    box_var = box_std_dev[0][0]**2
    all_var = all_std_dev[0][0]**2
    box_mean = box_mean[0][0]
    all_mean = all_mean[0][0]
    return (box_mean - all_mean)**2 / (box_var + all_var)


def mean_diff(grayscale, box=None):
    if box is None:
        box = utilities.get_box(grayscale, utilities.get_middle(grayscale), 100)
    box_mean, box_std_dev = cv2.meanStdDev(box)
    all_mean, all_std_dev = cv2.meanStdDev(grayscale)
    # print(abs(box_mean - all_mean), 70*box_std_dev**2)
    import math
    alpha = 0.01
    print(abs(box_mean - all_mean)[0][0]*alpha, (1-alpha)*box_std_dev[0][0]**2, 0*all_std_dev[0][0]**2)
    # return abs(box_mean - all_mean) - 70*box_std_dev**2# - 30*all_std_dev**2
    return alpha*abs(box_mean - all_mean)[0][0] - (1-alpha)**box_std_dev[0][0]**2 - 0*all_std_dev[0][0]**2


def ball_fitness_func(grayscale):
    return 1.0


class Transformer:
    def __init__(self, initial_playground_transformation_coefficients=None, initial_ball_transformations_coefficients=None, filename=None):
        # This isn't saved for now
        self.playground_fitness_func = mean_diff

        if filename is not None:
            #print("Loading")
            with open(filename, "r") as f:
                json_object = json.load(f)
                self.playground_transformations = json_object["playground_transformations"]
                self.ball_transformations = json_object["ball_transformations"]

        else:
            self.playground_transformations = initial_playground_transformation_coefficients
            self.ball_transformations = initial_ball_transformations_coefficients

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump({"playground_transformations": self.playground_transformations,
                       "ball_transformations": self.ball_transformations}, f)

    def get_playground_transformation(self, image):
        """
        Get the optimal transform for finding the playground
        """

        # Sort transformations according to their fitness for this image
        self.playground_transformations.sort(key=lambda transform: self.playground_fitness_func(utilities.transform_image(image, transform)))

        cur_best_transform = self.playground_transformations[-1]
        cur_best_fitness = self.playground_fitness_func(utilities.transform_image(image, cur_best_transform))

        # Perform a quick hill climber before returning
        optimization_method = "hill climber"
        # optimization_method = "gradient ascent"
        # optimization_method = None

        if optimization_method == "hill climber":
            step_size = 0.3
            max_iters = 4
            for idx in range(max_iters):
                # utilities.show(utilities.transform_image(image, cur_best_transform), time_ms=30)
                new_best = False
                for color_space, coefficient in cur_best_transform.items():
                    c1 = cur_best_transform.copy()
                    c2 = cur_best_transform.copy()

                    # If c1[i] is 0.0 we won't ever change the value unless we add a minimum value like here
                    c1[color_space] += max(0.05, step_size * c1[color_space])
                    c2[color_space] -= max(0.05, step_size * c2[color_space])

                    c1_fitness = self.playground_fitness_func(utilities.transform_image(image, c1))
                    c2_fitness = self.playground_fitness_func(utilities.transform_image(image, c2))

                    if c1_fitness > cur_best_fitness:
                        cur_best_transform = c1
                        cur_best_fitness = c1_fitness
                        new_best = True
                        # break

                    if c2_fitness > cur_best_fitness:
                        cur_best_transform = c2
                        cur_best_fitness = c2_fitness
                        new_best = True
                        # break

                transformed = utilities.transform_image(image, cur_best_transform)
                utilities.show(transformed, time_ms=30, text=str(self.playground_fitness_func(transformed)))

                if not new_best:
                    break

        elif optimization_method == "gradient ascent":
            step_size = 100
            max_iters = 10
            h = 0.01
            for idx in range(max_iters):
                highest = 0
                for key, coefficient in cur_best_transform.items():
                    highest = max(highest, abs(coefficient))

                for key in cur_best_transform:
                    cur_best_transform[key] /= highest

                gradient = {}
                # utilities.show(utilities.transform_image(image, cur_best_transform), time_ms=30)
                new_best = False
                for color_space, coefficient in cur_best_transform.items():
                    c1 = copy.deepcopy(cur_best_transform)
                    c2 = copy.deepcopy(cur_best_transform)

                    c1[color_space] += h/2
                    c2[color_space] -= h/2

                    c1_fitness = self.playground_fitness_func(utilities.transform_image(image, c1))
                    c2_fitness = self.playground_fitness_func(utilities.transform_image(image, c2))

                    gradient[color_space] = (c1_fitness - c2_fitness) / h

                # print(gradient)
                for color_space, coefficient in cur_best_transform.items():
                    cur_best_transform[color_space] = coefficient + gradient[color_space] * step_size

                print(self.playground_fitness_func(utilities.transform_image(image, cur_best_transform)))
                utilities.show(utilities.transform_image(image, cur_best_transform), time_ms=30)
        # Update the previous best
        if self.playground_transformations[-1] != cur_best_transform:
            print("Updating transform: {} to {}".format(self.playground_transformations[-1], cur_best_transform))
            self.playground_transformations[-1] = cur_best_transform

        return utilities.transform_image(image, cur_best_transform)

    @staticmethod
    def get_light_mask(image):
        # This function is very rough, and probably doesn't work very well
        grayscale = cv2.cvtColor(image.get_bgr(), cv2.COLOR_BGR2GRAY)
        grayscale = utilities.as_uint8(grayscale)
        grayscale[np.where(grayscale > 240)] = 255
        grayscale[np.where(grayscale < 255)] = 0
        return grayscale

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

    # t = Transformer([[-1.251020254026619, -2.43429755767094, 0.20662770269975134, 1.0668678186439797, 1.195385752636115, -9.782361764989172, 3.8060870336440566, 0.057165293306239345, 15.806092657763976, -1.3061166338412902]], [])
    # image = Image("2016-04-13_06:32:04.png")
    # utilities.show(image.get_bgr())
    # transform = t.get_playground_transformation(image)
    # utilities.show(transform)
    # exit(0)

    import networkcamera
    transformer = Transformer(filename="playground_transformer_state.json")
    #
    # t1 = Transformer()
    # t2 = Transformer()
    # t3 = Transformer()
    # t4 = Transformer()
    #
    # image = Image("/home/anders/UNIK4690/project/images/raspberry/south/2016-05-02_11:51:21.png")
    #
    # t1.get_playground_transformation(image)
    # t2.get_playground_transformation(image)
    # t3.get_playground_transformation(image)
    # t4.get_playground_transformation(image)
    #
    # exit(0)

    #{"ball_transformations": null, "playground_transformations": [{"lab_a": -0.3835904764611054, "ycrcb_y": 0.18121514971491337, "ycrcb_cn": -0.010627159328814776, "bgr_r": -0.7900661518043041, "bgr_b": 0.11399199903420945, "ycrcb_cr": -0.6889582901043134, "hsv_h": 0.9119966281808028, "hsv_v": -1.0, "bgr_g": 0.6797306701997627, "hsv_s": -0.24043657910513946, "lab_b": -0.28036275430352386, "lab_l": -0.16593829211255623}]}
    # with networkcamera.NetworkCamera("http://31.45.53.135:1337/raspberry_image.png") as cam:
    #     cam.set_resolution(1920, 1080)
    #     image = Image(image_data=cam.capture())
    #
    # cv2.imwrite("images/raspberry_1.png", image.get_bgr(np.uint8))
    #
    # #image = Image("images/raspberry_1.png")
    #
    # utilities.show_all(image)
    #
    # try:
    #     idx = 0
    #     while True:
    #         best_transform = transformer.get_playground_transformation(image)
    #         utilities.show(best_transform, text=str(idx), time_ms=30)
    #         idx += 1
    # except Exception as e:
    #     import traceback
    #     print(e)
    #     traceback.print_tb(e.__traceback__)
    #     transformer.save(filename="playground_transformer_state.json")
    # exit(0)
    #
    # import networkcamera, camera
    #
    # with networkcamera.NetworkCamera("http://31.45.53.135:1337/new_image.png") as cam:
    #     cam.set(camera.EXPOSURE, 10)
    #     while True:
    #         frame = cam.capture()
    #         image = Image(image_data=frame, color_normalization=True)
    #
    #         best_transform = transformer.get_playground_transformation(image)
    #         utilities.show(best_transform, time_ms=30)

    try:
        import os
        filenames = []
        for cur in os.walk(os.path.join(utilities.get_project_directory(), "images/raspberry/south/")):
            filenames = cur[2]
            break

        filenames.sort()

        for file in filenames:
            try:
                import datetime
                date = datetime.datetime.strptime(file, "%Y-%m-%d_%H:%M:%S.png")
                # if date < datetime.datetime(2016, 4, 13, 7, 5):
                if date < datetime.datetime(2016, 4, 12, 19, 0):
                    continue
                image = Image(os.path.join("images/raspberry/south/", file))
                utilities.show_all(image)
                # color_spaces = image.get_color_space_dict()
                #
                # def my_fitness(transform, box=None):
                #     if box is None:
                #         box = utilities.get_box(transform, utilities.get_middle(transform), 100)
                #     box_mean, box_std_dev = cv2.meanStdDev(box)
                #     all_mean, all_std_dev = cv2.meanStdDev(transform)
                #     print(abs(box_mean - all_mean), 1/(10000*box_std_dev**2))
                #     return abs(box_mean - all_mean) + 1/(10000*box_std_dev**2)
                #     return abs(box_mean - all_mean) - 7000*box_std_dev**2
                # c = [0] * len(color_spaces)
                # for idx, space in enumerate(color_spaces):
                #     fit = my_fitness(space)[0][0]
                #     c[idx] = fit
                # print(c)
                # transform = utilities.transform_image((bgr, hsv, lab, ycrcb), c)
                # utilities.show(bgr[:,:,0], text=image.filename)
                # utilities.show(transform, text=image.filename)
                # continue

            except FileNotFoundError:
                continue

            best_transform = transformer.get_playground_transformation(image)

            utilities.show(best_transform, time_ms=30, text=image.filename)
    except Exception as e:
        import traceback
        print(e)
        traceback.print_tb(e.__traceback__)
    finally:
        transformer.save("playground_transformer_state.json")
