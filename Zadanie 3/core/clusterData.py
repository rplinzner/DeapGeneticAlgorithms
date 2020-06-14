import numpy as np
from core.helpers import kmeans, clusterData
from scipy.spatial.distance import euclidean


class ClusterData:
    def __init__(self, numOfClusters, numOfDataInCluster, data):
        self.numOfClusters = numOfClusters
        self.numOfDataInCluster = numOfDataInCluster
        self.data = data

    def eval(self, individual):
        splitIndividual = np.array(individual).reshape(
            self.numOfClusters, self.numOfDataInCluster)

        clusterIdentifiers = kmeans(splitIndividual, self.data)

        clusteredData = clusterData(
            self.data, clusterIdentifiers, self.numOfClusters)

        averageDistances = np.zeros(len(splitIndividual))
        averageDensity = np.zeros(len(splitIndividual))

        closestDistances = np.zeros(
            (self.numOfClusters - 1) ** self.numOfClusters)

        index = -1

        for i in range(self.numOfClusters):
            numOfElemsInCluster = len(clusteredData[i])
            # operations within one cluster
            distancesSum = 0
            densitySum = 0
            if numOfElemsInCluster != 0:
                for el1 in range(numOfElemsInCluster):
                    densitySum += euclidean(clusteredData[i]
                                            [el1], splitIndividual[i])
                    for el2 in range(el1 + 1, numOfElemsInCluster):
                        distancesSum += euclidean(
                            clusteredData[i][el1], clusteredData[i][el2])
                averageDistances[i] = distancesSum / numOfElemsInCluster
                averageDensity[i] = densitySum / numOfElemsInCluster
                # closest elements
                for j in range(self.numOfClusters):
                    if i == j:
                        continue

                    numofElemsInSecond = len(clusteredData[j])
                    if (numofElemsInSecond == 0):
                        continue
                    index = index + 1
                    for ind in range(len(clusteredData[i])):
                        for ind1 in range(len(clusteredData[j])):
                            currentDistance = euclidean(
                                clusteredData[i][ind], clusteredData[j][ind1])
                            if closestDistances[index] == 0 or closestDistances[index] > currentDistance:
                                closestDistances[index] = currentDistance
        return sum(averageDistances), sum(averageDensity), sum(closestDistances)
