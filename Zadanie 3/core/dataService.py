import numpy as np
import random


class DataService:

    def __init__(self, filename, numberOfClusters, numberOfRows):
        self.filename = filename
        self.loadedData = []
        self.numberOfClusters = numberOfClusters
        self.numberOfRows = numberOfRows

    def load_data(self, skip_cols=[], skipRows=0, delimeter=','):
        max_rows = self.numberOfRows
        with open(self.filename) as f:
            numOfCols = len(f.readline().split(delimeter))

        use_cols = set(np.arange(0, numOfCols)) - set(skip_cols)
        use_cols = sorted(use_cols)

        self.loadedData = np.loadtxt(
            self.filename, delimiter=delimeter, usecols=use_cols, max_rows=max_rows, skiprows=skipRows)

    def getRandomIndividual(self):
        samples = np.array([])
        for index in range(self.numberOfClusters):
            samples = np.append(samples, random.choice(self.loadedData))
        return samples

    def group_data(self, assignments, groups):
        grouped = {}
        for i in range(groups):
            grouped.setdefault(i, [])

        for idx, val in enumerate(assignments):
            grouped[val].append(self.loadedData[idx])

        return grouped
