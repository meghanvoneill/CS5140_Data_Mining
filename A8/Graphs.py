import numpy as np
from sklearn import preprocessing
from scipy import linalg as LA
import random


def main():

    # A) Run each method (with t = 1024, q_0 = [1, 0, 0, . . ., 0]^T and t_0 = 100 when needed) and
    #    report the answers.
    M = np.loadtxt('M.csv', delimiter=',')
    t = 1024
    q_0 = np.zeros((1, M.shape[1]))
    q_0[0][0] = 1
    t_0 = 100

    q_star_1 = matrix_power_qstar(M, q_0)
    print('matrix_power_qstar: ')
    print(q_star_1)
    q_star_2 = state_propagation_qstar(M, q_0, t)
    print('state_propogation_qstar: ')
    print(q_star_2)
    q_star_3 = random_walk_qstar(M, t)
    print('random_walk_qstar: ')
    print(q_star_3)
    q_star_4 = eigen_analysis_qstar(M)
    print('eigen_analysis_qstar: ')
    print(q_star_4)

    # B) Rerun the Matrix Power and State Propagation techniques with q_0 = [0.1, 0.1, . . . , 0.1]^T.
    #    For what value of t is required to get as close to the true answer as the older initial state?
    q_0 = np.full((1, M.shape[1]), 0.1)

    q_star_5 = matrix_power_qstar(M, q_0)
    print('matrix_power_qstar: ')
    print(q_star_5)
    q_star_6 = state_propagation_qstar(M, q_0, t)
    print('state_propogation_qstar: ')
    print(q_star_6)

    # C) Explain at least one Pro and one Con of each approach. The Pro should explain a situation when
    #    it is the best option to use. The Con should explain why another approach may be better for
    #    some situation.

    # D) Is the Markov chain ergodic? Explain why or why not.

    # E) Each matrix M row and column represents a node of the graph, label these from 0 to 9 starting
    #    from the top and from the left. What nodes can be reached from node 4 in one step, and with
    #    what probabilities?

    return


# Matrix Power: Choose some large enough value t, and create M^t. Then apply q∗ = (M^t)q_0. There are
#               two ways to create M^t, first we can just let M^i+1 = M^i ∗ M, repeating this process t − 1
#               times. Alternatively, (for simplicity assume t is a power of 2), then in log_2(t) steps
#               create M^2i = M^i ∗ M^i.
def matrix_power_qstar(M, q_0):

    t = 100
    M_t = np.linalg.matrix_power(M, t)
    q_0 = q_0.reshape(-1, 1)
    q_star = M_t.dot(q_0)

    return q_star


# State Propagation: Iterate q_(i+1) = M ∗ q_i for some large enough number t iterations.
def state_propagation_qstar(M, q_0, t):
    q_i = q_0.reshape(-1, 1)

    for i in range(t):
        q_i = M.dot(q_i)

    return q_i


# Random Walk: Starting with a fixed state q_0 = [0,0,...,1,...,0,0]^T where there is only a 1 at the ith
#              entry, and then transition to a new state with only a 1 in the jth entry by choosing a new
#              location proportional to the values in the ith column of M. Iterate this some large number
#              t_0 of steps to get state q'_0. (This is the burn in period.)
#
#              Now make t new step starting at q'_0 and record the location after each step. Keep track of
#              how many times you have recorded each location and estimate q∗ as the normalized version
#              (recall ||q∗||_1 = 1) of the vector of these counts.
def random_walk_qstar(M, t, burn_in_size=5000):

    q_star = np.zeros(M.shape[1])
    i = 1

    # Burn in period:
    for m in range(burn_in_size):
        # Calculate the next state using Roulette Wheel Selection.
        current_node = M[:, i]
        random_num_to_match = random.uniform(0, 1)
        sum = 0
        for k in range(current_node.shape[0] + 1):
            if sum >= random_num_to_match:
                next_node_index = k - 1
                break
            sum += current_node[k,]
        i = next_node_index

    # Walk:
    for m in range(t):
        # Calculate the next state using Roulette Wheel Selection.
        current_node = M[:, i]
        random_num_to_match = random.uniform(0, 1)
        sum = 0
        for k in range(current_node.shape[0] + 1):
            if sum >= random_num_to_match:
                next_node_index = k - 1
                break
            sum += current_node[k, ]
        i = next_node_index
        q_star[i] += 1

    q_star = q_star.reshape(1, -1)
    normalized_q_star = preprocessing.normalize(q_star, 'l1')

    return normalized_q_star


# Eigen-Analysis: Compute LA.eig(M) and take the first eigenvector after it has been L1-normalized.
def eigen_analysis_qstar(M):

    eigenvalues, eigenvectors = LA.eig(M)
    normalized_eigenvectors = preprocessing.normalize(eigenvectors.real, 'l1')
    eigenvector_0 = normalized_eigenvectors[:, 0]

    return eigenvector_0


if __name__ == "__main__":
    main()
