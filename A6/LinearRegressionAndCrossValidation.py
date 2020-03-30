import numpy as np
import numpy.linalg as la
import random


# We will find coefficients alpha to estimate: X * alpha ≈ y, using the provided data sets X and y.
#   We will compare two approaches, least squares and ridge regression.
def main():

    X = np.loadtxt('X.csv', delimiter=',')
    y = np.loadtxt('y.csv', delimiter=',')

    s_options = [0.2, 0.4, 0.8, 1.0, 1.2, 1.4, 1.6]
    s = random.choice(s_options)

    alpha = la.inv(X.T @ X) @ X.T @ y.T
    alphas = la.inv(X.T @ X + s * np.identity(50)) @ X.T @ y.T

    print(alpha)
    print(alphas)

    # A: Solve for the coefficients alpha(or alphas) using Least Squares and Ridge Regression with
    #       s ∈ {0.2, 0.4, 0.8, 1.0, 1.2, 1.4, 1.6} (i.e. s will take on one of those 7 values each
    #       time you try, say obtaining alpha04 for s = 0.4). For each set of coefficients, report the
    #       error in the estimate y_hat of y as norm(y - X * alpha, 2).

    X1 = X[:66, :]
    Y1 = y[:66]
    X2 = X[33:, :]
    Y2 = y[33:]
    X3 = np.vstack((X[:33, :], X[66:, :]))
    Y3 = np.vstack(y[:33], y[66:])

    # Report the error in the estimate of y_hat of y as: norm(y - X * alpha, 2).
    # error = np.norm(y - X * alpha, 2)

    # B: Create three row-subsets of X and Y. Repeat the above procedure on these subsets and
    #       cross-validate the solution on the remainder of X and Y. Specifically, learn the coefficients
    #       alpha using, say, X1 and Y1 and then measure np.norm(Y[66:] - X[66:, :] @ alpha, 2).

    # C: Which approach works best (averaging the results from the three subsets): Least Squares, or
    #       for which value of s using Ridge Regression?

    # D: Use the same 3 test / train splits, taking their average errors, to estimate the average
    #       squared error on each predicted data point.
    #   What is problematic about the above estimate, especially for the best performing parameter
    #       value s?

    # E: Even circumventing the issue raised in part D, what assumptions about how the data set (X, y)
    #       is generated are needed in an assessment based on cross-validation?

    # 2: Consider a linear equation W = M * S where M is a measurement matrix filled with random values
    #       {-1, 0, +1} (although now that they are there, they are no longer random), and W is the
    #       output of the sparse signal S when measured by M.
    #   Use Matching Pursuit (as described in the book as Algorithm 5.5.1 to recover the non-zero entries
    #       S. Record the order in which you find each entry and the residual vector after each step.


def lin_regression():
    return


def cross_validation():
    return


if __name__ == '__main__':
    main()

