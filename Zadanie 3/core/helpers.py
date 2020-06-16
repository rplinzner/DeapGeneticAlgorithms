import numpy as np
from scipy.spatial.distance import euclidean
import random
import matplotlib.pyplot as plt


def kmeans(centroids, dataToCluster):
    # Assign all data to cluster 0
    clusteredData = np.zeros(len(dataToCluster), dtype=int)

    for index in range(len(clusteredData)):
        for cluster in range(len(centroids)):
            if euclidean(dataToCluster[index], centroids[cluster]) < euclidean(dataToCluster[index], centroids[clusteredData[index]]):
                clusteredData[index] = cluster
    return clusteredData


def clusterData(data, clusterIndexes, numOfClusters):
    ret = [[] for i in range(numOfClusters)]

    for index, clusterId in enumerate(clusterIndexes):
        ret[clusterId].append(data[index])
    return ret


def plotDataWithCentroids(individual, numOfClusters, numOfElemsInCluster, data):
    #
    centroids = np.array(individual).reshape(
        numOfClusters, numOfElemsInCluster)

    assignments = kmeans(centroids, data)

    print("Assigned data:")
    print(assignments)

    plots = [[] for k in range(len(centroids))]

    for i in range(len(centroids)):
        for centr, value in enumerate(assignments):
            if i == value:
                plots[i].append(data[centr])

    colors = ['red', 'green', 'blue']

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    for cluster in range(len(plots)):

        x = []
        y = []
        z = []

        if len(plots[cluster]) > 0:
            x = np.array(plots[cluster])[:, 0]
            y = np.array(plots[cluster])[:, 1]
            z = np.array(plots[cluster])[:, 2]

        xCentroid = centroids[cluster][0]
        yCentroid = centroids[cluster][1]
        zCentroid = centroids[cluster][2]

        ax.scatter(x, y, z, c=colors[cluster], marker='x', s=25)

        ax.scatter(xCentroid, yCentroid, zCentroid,
                   c='black', marker='.', s=200)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
