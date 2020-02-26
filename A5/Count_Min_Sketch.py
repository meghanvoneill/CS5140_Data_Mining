import numpy as np


def main():

    file_name_1 = 'S1.txt'
    file_name_2 = 'S2.txt'

    output = count_min_sketch(file_name_1, 10)
    print('output: ' + str(output))

    return


def count_min_sketch(file_name, k, t):

    count_array = np.array()

    # While s has data
    with open(file_name, 'r') as s:

        for line in iter(s.readline, ''):

            for i in line:

                for j in range(0, t):

                    count_array[i][j] = count_array[i][j] + 1



    return count_array


if __name__ == '__main__':
    main()