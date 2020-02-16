# coding=utf-8
import numpy as npy
import matplotlib.pyplot as plt
import math
import random


# Author: Meghan V. O'Neill
# For Asmt 4: Clustering
# Data Mining, Spring 2020, Professor Phillips, University of Utah

# Overview
#   In this assignment you will explore clustering: hierarchical and point-assignment. You will also experiment
#   with high dimensional data.
#   You will use three data sets for this assignment:
#       • http://www.cs.utah.edu/ ̃jeffp/teaching/cs5140/A4/C1.txt
#       • http://www.cs.utah.edu/ ̃jeffp/teaching/cs5140/A4/C2.txt
#       • http://www.cs.utah.edu/ ̃jeffp/teaching/cs5140/A4/C3.txt
#   These data sets all have the following format. Each line is a data point. The lines have either 3 or 6 tab
#   separated items. The first one is an integer describing the index of the points. The next 2 (or 5 for C3)
#   are the coordinates of the data point. C1 and C2 are in 2 dimensions, and C3 is in 5 dimensions. C1 should
#   have n=19 points, C2 should have n=1040 points, and C3 should have n=1000 points. We will always measure
#   distance with Euclidean distance.
def main():
    # 1.A: Run all hierarchical clustering variants on data set C1.txt until there are k = 4 clusters,
    #      and report the results as sets. It may be useful to do this pictorially.

    C1_data = {}
    C1_file = 'C1.txt'
    C1_xlist = []
    C1_ylist = []

    with open(C1_file, 'r') as f:
        line = f.readline()
        while line:
            words_in_line = line.split()
            x = float(words_in_line[1])
            y = float(words_in_line[2])
            C1_data[int(words_in_line[0])] = (x, y)
            C1_xlist.append(x)
            C1_ylist.append(y)

            line = f.readline()

    # Plot data without alteration.
    # plt.scatter(C1_xlist, C1_ylist, marker='o')
    # plt.ylabel('y')
    # plt.xlabel('x')
    # plt.title('C1')
    # plt.show()

    ##########################################################################################################
    #### Single-Link Distance ####
    ##########################################################################################################
    # k = 4
    # threshold = 3.7
    # clusters_Q1A = {}
    # next_ID = 0
    #
    # # Add all points as their own clusters to the clusters dictionary.
    # for i in range(len(C1_xlist)):
    #     point = {'x': C1_xlist[i], 'y': C1_ylist[i]}
    #     points = {'0': point}
    #     cluster = {'points': points}
    #     clusters_Q1A[next_ID] = cluster
    #     next_ID += 1
    #
    # # While we don't have k clusters
    # while len(clusters_Q1A.keys()) > k:
    #
    #     # Randomly choose starting cluster
    #     S_i_key = random.choice(list(clusters_Q1A))
    #     S_i = clusters_Q1A[S_i_key]
    #
    #     # For every other cluster
    #     for S_j_key in clusters_Q1A.keys():
    #         # If S_i != S_j
    #         if S_i_key != S_j_key:
    #             # Find the closest pair
    #             closest = single_link_distance(S_i, clusters_Q1A[S_j_key])
    #             # If two given clusters, S_i and S_j, are close enough (threshold)
    #             if closest['min_val'] <= threshold:
    #                 # Merge S_i and S_j into a single cluster
    #                 next_ID = len(S_i['points'].keys())
    #                 print("next_ID: " + str(next_ID))
    #                 print("merging " + str(S_i) + " and " + str(clusters_Q1A[S_j_key]['points']))
    #                 for s_j in clusters_Q1A[S_j_key]['points']:
    #                     print(clusters_Q1A[S_j_key]['points'])
    #                     print("s_j: " + str(s_j))
    #                     S_i['points'][next_ID] = clusters_Q1A[S_j_key]['points'][s_j]
    #                     next_ID += 1
    #                     print("new next_ID: " + str(next_ID))
    #                 del clusters_Q1A[S_j_key]
    #                 print("result: " + str(S_i))
    #
    # print('Found ' + str(k) + ' clusters!')

    #print_Q1_data(clusters_Q1A, 'C1 - Single-Link Distance')

    ##########################################################################################################
    #### Complete-Link Distance ####
    ##########################################################################################################
    # k = 4
    # threshold = 7
    # clusters_Q1B = {}
    # next_ID = 0
    #
    # # Add all points as their own clusters to the clusters dictionary.
    # for i in range(len(C1_xlist)):
    #     point = {'x': C1_xlist[i], 'y': C1_ylist[i]}
    #     points = {'0': point}
    #     cluster = {'points': points}
    #     clusters_Q1B[next_ID] = cluster
    #     next_ID += 1
    #
    # # While we don't have k clusters
    # while len(clusters_Q1B.keys()) > k:
    #
    #     # Randomly choose starting cluster
    #     S_i_key = random.choice(list(clusters_Q1B))
    #     S_i = clusters_Q1B[S_i_key]
    #
    #     # For every other cluster
    #     for S_j_key in clusters_Q1B.keys():
    #         # If S_i != S_j
    #         if S_i_key != S_j_key:
    #             # Find the closest pair
    #             closest = complete_link_distance(S_i, clusters_Q1B[S_j_key])
    #             # If two given clusters, S_i and S_j, are close enough (threshold)
    #             if closest['max_val'] <= threshold:
    #                 # Merge S_i and S_j into a single cluster
    #                 next_ID = len(S_i['points'].keys())
    #                 print("next_ID: " + str(next_ID))
    #                 print("merging " + str(S_i) + " and " + str(clusters_Q1B[S_j_key]['points']))
    #                 for s_j in clusters_Q1B[S_j_key]['points']:
    #                     print(clusters_Q1B[S_j_key]['points'])
    #                     print("s_j: " + str(s_j))
    #                     S_i['points'][next_ID] = clusters_Q1B[S_j_key]['points'][s_j]
    #                     next_ID += 1
    #                     print("new next_ID: " + str(next_ID))
    #                 del clusters_Q1B[S_j_key]
    #                 print("result: " + str(S_i))
    #
    # print('Found ' + str(k) + ' clusters!')

    #print_Q1_data(clusters_Q1B, 'C1 - Complete-Link Distance')

    ##########################################################################################################
    #### Mean-Link Distance ####
    ##########################################################################################################
    k = 4
    threshold = 5.5
    clusters_Q1C = {}
    next_ID = 0

    # Add all points as their own clusters to the clusters dictionary.
    for i in range(len(C1_xlist)):
        point = {'x': C1_xlist[i], 'y': C1_ylist[i]}
        points = {'0': point}
        cluster = {'points': points, 'mean': point}
        clusters_Q1C[next_ID] = cluster
        next_ID += 1

    # While we don't have k clusters
    while len(clusters_Q1C.keys()) > k:

        # Randomly choose starting cluster
        S_i_key = random.choice(list(clusters_Q1C))
        S_i = clusters_Q1C[S_i_key]

        # For every other cluster
        for S_j_key in clusters_Q1C.keys():
            # If S_i != S_j
            if S_i_key != S_j_key:
                # Find the closest pair
                distance = abs(mean_link_distance(S_i, clusters_Q1C[S_j_key]))
                # If two given clusters, S_i and S_j, are close enough (threshold)
                if distance <= threshold:
                    # Merge S_i and S_j into a single cluster
                    next_ID = len(S_i['points'].keys())
                    print("merging " + str(S_i) + " and " + str(clusters_Q1C[S_j_key]['points']))
                    for s_j in clusters_Q1C[S_j_key]['points']:
                        S_i['points'][next_ID] = clusters_Q1C[S_j_key]['points'][s_j]
                        next_ID += 1
                    cluster_mean(S_i)
                    del clusters_Q1C[S_j_key]

    print('Found ' + str(k) + ' clusters!')

    print_Q1_data(clusters_Q1C, 'C1 - Mean-Link Distance')


    # 1.B: Which variant did the best job, and which was the easiest to compute (think if the data was
    #      much larger)? Explain your answers.

    # 2: Assignment-based clustering works by assigning every point x ∈ X to the closest cluster centers C.
    #    Let φ_C : X → C be this assignment map so that φ_C(x) = arg min_{c∈C}d(x, c). All points that map
    #    to the same cluster center are in the same cluster.

    # 2.A: Run Gonzalez and k-Means++ on data set C2.txt for k = 3. To avoid too much variation
    #      in the results, choose c1 as the point with index 1.
    #      Report the centers and the subsets (as pictures) for Gonzalez.
    #      Report:
    #           • the 3-center cost maxx∈X d(x, φC (x)) and
    #           • the 3-means cost 􏰁 1 􏰀 (d(x, φC (x)))2 |X| x∈X
    #      (Note this has been normalized so easy to compare to 3-center cost)

    # 2.B: For k-Means++, the algorithm is randomized, so you will need to report the variation in this algorithm.
    #      Run it several trials (at least 20) and plot the cumulative density function of the 3-means cost. Also
    #      report what fraction of the time the subsets are the same as the result from Gonzalez.

    # plt.plot(new_y_axis)
    # plt.title('m = 300')
    # plt.ylabel('fraction of experiments that succeeded')
    # plt.xlabel('k')
    # plt.show()

    # 2.C: Recall that Lloyd’s algorithm for k-means clustering starts with a set of k centers C and runs as
    #      described in Algorithm 8.3.1 (in M4D).
    #           1. Run Lloyds Algorithm with C initially with points indexed {1,2,3}. Report the final subset
    #              and the 3-means cost.
    #           2. Run Lloyds Algorithm with C initially as the output of Gonzalez above. Report the final
    #              subset and the 3-means cost.
    #           3. Run Lloyds Algorithm with C initially as the output of each run of k-Means++ above. Plot a
    #              cumulative density function of the 3-means cost. Also report the fraction of the trials that
    #              the subsets are the same as the input (where the input is the result of k-Means++).

    return


# Euclidean Distance:
def euclidean_dist(x_1, y_1, x_2, y_2):
    take_square_root = math.pow((x_1 - x_2), 2) + math.pow((y_1 - y_2), 2)
    d = math.sqrt(take_square_root)

    return d


# Single-Link:
#   measures the shortest link d(S1, S2) = min_{(s1 ,s2)∈ S1×S2}||s1 − s2||_2.
def single_link_distance(S1, S2):

    min = {}
    min['min_val'] = float('inf')
    min['s1'] = float('inf')
    min['s2'] = float('inf')

    for s1 in S1['points'].keys():
        for s2 in S2['points'].keys():
            sub = euclidean_dist(S1['points'][s1]['x'], S1['points'][s1]['y'],
                                 S2['points'][s2]['x'], S2['points'][s2]['y'])
            if sub < min['min_val']:
                min['min_val'] = sub
                min['s1'] = s1
                min['s2'] = s2

    return min


# Complete-Link:
#   measures the longest link d(S1, S2) = max_{(s1 ,s2)∈ S1×S2}||s1 − s2||_2.
def complete_link_distance(S1, S2):

    max = {}
    max['max_val'] = -float('inf')
    max['s1'] = -float('inf')
    max['s2'] = -float('inf')

    for s1 in S1['points'].keys():
        for s2 in S2['points'].keys():
            sub = euclidean_dist(S1['points'][s1]['x'], S1['points'][s1]['y'],
                                 S2['points'][s2]['x'], S2['points'][s2]['y'])
            if sub > max['max_val']:
                max['max_val'] = sub
                max['s1'] = s1
                max['s2'] = s2

    return max


def cluster_mean(cluster):

    summation_x = 0
    summation_y = 0

    for p in cluster['points'].keys():
        summation_x += cluster['points'][p]['x']
        summation_y += cluster['points'][p]['y']

    print('len: ' + str(len(cluster['points'].keys())))

    mean_x = (1 / float(len(cluster['points'].keys()))) * summation_x
    mean_y = (1 / float(len(cluster['points'].keys()))) * summation_y

    mean = {'x': mean_x, 'y': mean_y}
    cluster['mean'] = mean

    print('cluster: ' + str(cluster))
    print('mean: ' + str(mean))
    return


# Mean-Link:
#   a_1 = (1/|S_1|) * sum_{s ∈ S_1}(s)
#   a_2 = (1/|S_2|) * sum_{s ∈ S_2}(s)
#   d(S1, S2) = ||a1 − a2||_2
def mean_link_distance(S1, S2):

    distance = euclidean_dist(S1['mean']['x'], S1['mean']['y'],
                         S2['mean']['x'], S2['mean']['y'])

    print('S1: ' + str(S1))
    print('S1 mean: ' + str(S1['mean']['x']) + ' ' + str(S1['mean']['y']))
    print('S2: ' + str(S2))
    print('S2 mean: ' + str(S2['mean']['x']) + ' ' + str(S2['mean']['y']))
    print('distance: ' + str(distance))

    return distance


def gonzalez_alg(data, k):
    # Choose c_1 as the point with index 1.
    return


def k_means_plusplus(data, k):
    return


def lloyds_alg(data, k, C):
    return


def print_Q1_data(clusters, title):

    cluster_1_x = []
    cluster_2_x = []
    cluster_3_x = []
    cluster_4_x = []
    cluster_1_y = []
    cluster_2_y = []
    cluster_3_y = []
    cluster_4_y = []
    count = 0
    colors = ['#39A2AE', '#CC0000', '#BADA55', '#F1C300', '#4E004F', '#3B0056']

    for cluster in clusters:
        if count == 0:
            for point in clusters[cluster]['points']:
                cluster_1_x.append(clusters[cluster]['points'][point]['x'])
                cluster_1_y.append(clusters[cluster]['points'][point]['y'])
            count += 1
        elif count == 1:
            for point in clusters[cluster]['points']:
                cluster_2_x.append(clusters[cluster]['points'][point]['x'])
                cluster_2_y.append(clusters[cluster]['points'][point]['y'])
            count += 1
        elif count == 2:
            for point in clusters[cluster]['points']:
                cluster_3_x.append(clusters[cluster]['points'][point]['x'])
                cluster_3_y.append(clusters[cluster]['points'][point]['y'])
            count += 1
        elif count == 3:
            for point in clusters[cluster]['points']:
                cluster_4_x.append(clusters[cluster]['points'][point]['x'])
                cluster_4_y.append(clusters[cluster]['points'][point]['y'])

    plt.scatter(cluster_1_x, cluster_1_y, color=colors[3], marker='o')
    plt.scatter(cluster_2_x, cluster_2_y, color=colors[1], marker='o')
    plt.scatter(cluster_3_x, cluster_3_y, color=colors[2], marker='o')
    plt.scatter(cluster_4_x, cluster_4_y, color=colors[0], marker='o')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title(title)
    plt.show()

    return


if __name__ == '__main__':
    main()
