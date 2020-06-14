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


def plot_data_set(cluster_centers, numOfClusters, numOfElemsInCluster, data):
    cluster_centers = np.array(cluster_centers).reshape(
        numOfClusters, numOfElemsInCluster)
    assignments = kmeans(cluster_centers, data)
    print(assignments)
    plots = [[] for k in range(len(cluster_centers))]
    for i in range(len(cluster_centers)):
        for idx, val in enumerate(assignments):
            if i == val:
                plots[i].append(data.data_set[idx])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'orange']
    for cluster in range(len(plots)):
        if len(plots[cluster]) > 0:
            x = np.array(plots[cluster])[:, 0]
            y = np.array(plots[cluster])[:, 1]
            z = np.array(plots[cluster])[:, 2]
        else:
            x = []
            y = []
            z = []
        xc = cluster_centers[cluster][0]
        yc = cluster_centers[cluster][1]
        zc = cluster_centers[cluster][2]

        ax.scatter(x, y, z, c=colors[cluster], marker='.', s=100)
        ax.scatter(xc, yc, zc, c='black', marker='.', s=250)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
