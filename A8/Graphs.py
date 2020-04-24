import numpy as np
from scipy import linalg as LA



def main():

    # A) Run each method (with t = 1024, q_0 = [1, 0, 0, . . ., 0]^T and t_0 = 100 when needed) and
    #    report the answers.
    M = np.loadtxt('M.csv', delimiter=',')
    t = 1024
    q_0 = np.zeros((1, M.shape[1]))
    q_0[0][0] = 1
    t_0 = 100

    matrix_power_qstar()
    state_propagation_qstar()
    random_walk_qstar()
    eigen_analysis_qstar(M)

    # B) Rerun the Matrix Power and State Propagation techniques with q_0 = [0.1, 0.1, . . . , 0.1]^T.
    #    For what value of t is required to get as close to the true answer as the older initial state?
    q_0 = np.full((1, M.shape[1]), 0.1)

    matrix_power_qstar()
    state_propagation_qstar()

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
def matrix_power_qstar():

    return


# State Propagation: Iterate q_(i+1) = M ∗ q_i for some large enough number t iterations.
def state_propagation_qstar():

    return


# Random Walk: Starting with a fixed state q_0 = [0,0,...,1,...,0,0]^T where there is only a 1 at the ith
#              entry, and then transition to a new state with only a 1 in the jth entry by choosing a new
#              location proportional to the values in the ith column of M. Iterate this some large number
#              t_0 of steps to get state q'_0. (This is the burn in period.)
#
#              Now make t new step starting at q'_0 and record the location after each step. Keep track of
#              how many times you have recorded each location and estimate q∗ as the normalized version
#              (recall ||q∗||_1 = 1) of the vector of these counts.
def random_walk_qstar():

    return


# Eigen-Analysis: Compute LA.eig(M) and take the first eigenvector after it has been L1-normalized.
def eigen_analysis_qstar(M):

    eig = LA.eig(M)
    print(eig)

    return


if __name__ == "__main__":
    main()
