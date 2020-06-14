import numpy as np
import matplotlib.pyplot as plt
import random


class DataService:

    def __init__(self, filename, numberOfClusters):
        self.filename = filename
        self.data_set = []
        self.n_cols = 0
        self.numberOfClusters = numberOfClusters

    def load_data(self, skip_cols=[], delimeter=',', max_rows=60):
        with open(self.filename) as f:
            self.n_cols = len(f.readline().split(delimeter))

        use_cols = set(np.arange(0, self.n_cols)) - set(skip_cols)
        use_cols = sorted(use_cols)

        self.data_set = np.loadtxt(
            self.filename, delimiter=delimeter, usecols=use_cols, max_rows=max_rows)

    def visualize_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        plot_data = np.empty((self.n_cols, len(self.data_set)))

        for i in range(0, self.n_cols - 1):
            plot_data[i] = self.data_set[:, i]

        ax.scatter(plot_data[0], plot_data[1],
                   plot_data[2], c='black', marker='.', s=100)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()

    def group_data(self, assignments, groups):
        grouped = {}
        for i in range(groups):
            grouped.setdefault(i, [])

        for idx, val in enumerate(assignments):
            grouped[val].append(self.data_set[idx])

        return grouped

    def getRandomIndividual(self):
        samples = np.array([])
        for index in range(self.numberOfClusters):
            samples = np.append(samples, random.choice(self.data_set))
        return samples
