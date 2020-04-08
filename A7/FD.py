import numpy as np
import math
from scipy import linalg as LA


def main():

    A = np.loadtxt('A.csv', delimiter=',')

    # 1.A
    svd_norm(A, 10)

    # 1.B
    # find_smallest_k_10(A)

    # 2.A
    l = 1
    #B = freq_dir(A, l)
    #error_max = LA.norm(A.T @ A - B.T @ B)
    #A2F = LA.norm(A, 'fro')^2

    #print(str(error_max))
    #print(str(A2F))


def freq_dir(A,l):
    r = A.shape[0]
    c = A.shape[1]
    B = np.zeros([l*2, c])
    B[:l-1, :] = A[:l-1,:]
    zerorows = l + 1

    for i in range(l-1,r):
        """
        implement the algorithm 16.2.1 in L16 MatrixSketching in Data Mining course webpage
          insert ith row into a zero-value row of the mat_b
          if B has no zero-valued rows (can be kept track with counter) then:
            U,S,V = svd(mat_b)  using  U,S,V = np.linalg.svd(mat_b,full_matrices = False)
            ...
            procedure same as the algorithm 16.2.1
            ...
        """

    return B


def svd_norm(A, k_max):
    U, s, Vt = LA.svd(A, full_matrices=False)
    S = np.diag(s)

    for k in range(0, k_max):
        Uk = U[:, :k]
        Sk = S[:k, :k]
        Vtk = Vt[:k, :]
        Ak = Uk @ Sk @ Vtk
        print('k: ' + str(k))
        print(str(LA.norm(A-Ak, 2)))

    return


def find_smallest_k_10(A):
    U, s, Vt = LA.svd(A, full_matrices=False)
    S = np.diag(s)
    small_k_found = False
    k = 0
    A_norm = LA.norm(A, 2)
    smallest_k = math.inf

    #while small_k_found is False:
    for k in range(50):
        Uk = U[:, :k]
        Sk = S[:k, :k]
        Vtk = Vt[:k, :]
        Ak = Uk @ Sk @ Vtk

        A_minus_Ak_norm = LA.norm(A-Ak, 2)

        print('k: ' + str(k))
        print(A_norm)
        print(str(A_minus_Ak_norm))

        if A_minus_Ak_norm < .10 * A_norm:
            if k < smallest_k:
                smallest_k = k
            small_k_found = True

        k += 1
    print('smallest k: ' + str(smallest_k))

    return


if __name__ == "__main__":
    main()