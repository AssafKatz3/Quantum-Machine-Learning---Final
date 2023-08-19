import numpy as np
import pickle
import os

class InitPointGenerator:
    def __init__(self, p, n, file_name='init_point_vectors.pkl'):
        self.p = p
        self.n = n
        self.file_name = file_name

    def generate_shuffled_init_point(self):
        total_variables = 2 * self.p
        init_points = []

        for _ in range(self.n):
            beta_range = np.linspace(0, 2 * np.pi, total_variables // 2)
            gamma_range = np.linspace(0, 2 * np.pi, total_variables // 2)

            np.random.shuffle(beta_range)
            np.random.shuffle(gamma_range)

            init_point = np.concatenate((beta_range, gamma_range))
            init_points.append(init_point)

        return init_points

    def generate_and_save_init_points(self):
        init_point_vectors = self.generate_shuffled_init_point()
        with open(self.file_name, 'wb') as file:
            pickle.dump(init_point_vectors, file)
        return init_point_vectors

    def load_init_points(self):
        with open(self.file_name, 'rb') as file:
            loaded_init_point_vectors = pickle.load(file)
        return loaded_init_point_vectors
