#!/usr/bin/python3
import random
import numpy as np
import cv2


root_path = "/home/anders/UNIK4690/project/"
# root_path = ""

class EvoAlg:
    def __init__(self, mutation_rate, mu, sigma, transform_image, population_size, individual_size, fitness_func,
                 parent_selection_pressure, to_be_killed_selection_pressure):
        self.mutation_rate = mutation_rate
        self.mu = mu
        self.sigma = sigma
        self.transform_image = transform_image
        self.n = individual_size
        self.fitness = {}
        self.fitness_func = fitness_func
        self.parent_selection_pressure = parent_selection_pressure
        self.to_be_killed_selection_pressure = to_be_killed_selection_pressure

        self.population = self._init_population(population_size)
        self.elite = self.population[-1]

    def _init_population(self, population_size):
        population = []
        # Initialize population
        for i in range(population_size):
            individual = [0]*self.n
            for j in range(self.n):
                individual[j] = random.gauss(self.mu, self.sigma)
            population.append(np.array(individual))
        population.sort(key=lambda x: self._get_fitness(x))
        return population

    def _get_fitness(self, individual):
        hashable = tuple(individual)
        if hashable not in self.fitness:
            self.fitness[hashable] = self.fitness_func(self.transform_image, individual)
        return self.fitness[hashable]

    def _mutate(self, individual):
        mutation = [0]*self.n
        mutation_prob = self.mutation_rate / self.n

        for i in range(self.n):
            if random.random() <= mutation_prob:
                mutation[i] = random.gauss(self.mu, self.sigma)

        return individual + np.array(mutation)

    def _recombination(self, parent1, parent2):
        return (parent1 + parent2) / 2

    def _select_parent(self):
        parents = self.population + [self.elite]
        parents.sort(key=lambda x: self._get_fitness(x))
        fitnesses = [self._get_fitness(i)**self.parent_selection_pressure for i in parents]
        tot_fitness = sum(fitnesses)
        num = random.random() * tot_fitness

        idx = 0
        cur = 0
        while cur < num:
            cur += fitnesses[idx]
            idx += 1
        return parents[idx-1]

    def _select_to_be_killed(self):
        self.population.sort(key=lambda x: self._get_fitness(x))
        fitnesses = [self._get_fitness(i)**self.to_be_killed_selection_pressure for i in self.population]
        highest = fitnesses[-1]
        fitnesses = [highest - i + 1 for i in fitnesses]
        tot_fitness = sum(fitnesses)
        num = tot_fitness * random.random()

        idx = 0
        cur = 0
        while cur < num:
            cur += fitnesses[idx]
            idx += 1
        return idx-1

    def run(self, max_iter=1000):
        i = 0

        try:
            while max_iter is None or i < max_iter:
                p1, p2 = self._select_parent(), self._select_parent()
                child = self._recombination(p1, p2)
                child = self._mutate(child)
                to_kill = self._select_to_be_killed()
                self.population[to_kill] = child

                self.population.sort(key=lambda x: self._get_fitness(x))

                self.elite = self.population[-1] if self._get_fitness(self.population[-1]) > self._get_fitness(self.elite) else self.elite

                print(self.elite, self._get_fitness(self.elite))
                i += 1

        except Exception as e:
            print("Caught exception: {}".format(e))
            print("Stopping EvoAlg.")

        finally:
            return self.elite


if __name__ == "__main__":

    def transform_image(img_spaces, vec):
        #from timeit import default_timer as timer
        #start = timer()

        img, hsv, lab, ycrcb = img_spaces

        transformed = np.zeros(img.shape[:2])
        if True:
            idx = 0
            b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
            transformed += vec[idx] * b
            idx += 1
            transformed += vec[idx] * g
            idx += 1
            transformed += vec[idx] * r
            idx += 1

            h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
            transformed += vec[idx] * h
            idx += 1
            transformed += vec[idx] * s
            idx += 1
            transformed += vec[idx] * v
            idx += 1

            l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
            transformed += vec[idx] * l
            idx += 1
            transformed += vec[idx] * a
            idx += 1
            transformed += vec[idx] * b
            idx += 1

            y, cr, cb = ycrcb[:,:,0], ycrcb[:,:,1], ycrcb[:,:,2]
            transformed += vec[idx] * y
            idx += 1
            transformed += vec[idx] * cr
            idx += 1
            transformed += vec[idx] * cb
            idx += 1


            # Normalization
            res = transformed - np.amin(transformed)
            res /= np.amax(res)

        else:
            color_spaces = []
            b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
            color_spaces += [b, g, r]

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
            color_spaces += [h, s, v]

            #lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            #l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
            #color_spaces += [l, a, b]

            #ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            #y, cr, cb = ycrcb[:,:,0], ycrcb[:,:,1], ycrcb[:,:,2]
            #color_spaces += [y, cr, cb]

            color_spaces = np.array(color_spaces)

            res = np.dot(color_spaces.transpose(), vec).transpose().reshape(img.shape[:2])
            # Normalization
            res -= np.amin(res)
            res /= np.amax(res)

        #print("Time used: {}s".format((timer() - start)))
        return res


    #img = cv2.imread("images/microsoft_cam/24h/south/2016-04-12_16:19:04.png")
    #transformed = transform_image(img, np.array([1,1,1]))
    #cv2.imshow("test", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    #cv2.waitKey(0)
    #cv2.imshow("test", transformed)
    #cv2.waitKey(0)
    #exit(0)

    import fitness
    fitness_func = fitness.create_fitness_function_v1(root_path+"images/microsoft_cam/24h/south/")
    alg = EvoAlg(mutation_rate=1, mu=0, sigma=1, transform_image=transform_image, population_size=20, individual_size=12, fitness_func=fitness_func,
                 parent_selection_pressure=1.0, to_be_killed_selection_pressure=1.0)
    best = alg.run(None)

    import os
    filenames = []
    for cur in os.walk(root_path+"images/microsoft_cam/24h/south/"):
        filenames = cur[2]
        break

    filenames.sort()

    for file in filenames:
        if file[-3:] == 'png':
            img = cv2.imread(root_path+"images/microsoft_cam/24h/south/" + file)
            cv2.putText(img, file, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            transformed = transform_image((img, hsv, lab, ycrcb), best)
            cv2.imshow("test", transformed)
            cv2.waitKey(30)

# Works ok: [ 0.36534565 -0.73206701 -1.22681424  0.01103986 -0.4275835   0.32282959 0.5432064   0.08782472 -0.64280642  1.24526084 -0.7005395  -0.24566722]
# b, g, r, h, s, v, l, a, b, y, cr, cb
