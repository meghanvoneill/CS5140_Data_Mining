

def main():

    file_name_1 = 'S1.txt'
    file_name_2 = 'S2.txt'

    output = misra_gries_algorithm(file_name_2, 9)
    print('output: ' + str(output))

    return


def misra_gries_algorithm(file_name, k):

    count_array = {}

    # While s has data
    with open(file_name, 'r') as s:

        for line in iter(s.readline, ''):

            for i in line:
                a_i = i

                # If the value in the stream is already a label
                if a_i in count_array.keys():
                    count_array[i] = count_array[i] + 1
                # Else if the amount of labels is less than k - 1
                elif len(count_array.keys()) < k - 1:
                    count_array[i] = 1
                # Else decrement every label
                else:
                    for label in count_array.keys():
                        count_array[label] = count_array[label] - 1
                        # If decrementing the label dropped the label's count to 0, delete the label
                        if count_array[label] == 0:
                            del count_array[label]

    return count_array


def count_min_sketch(file_name, k):

    count_array = {}

    # While s has data
    with open(file_name, 'r') as s:

        for line in iter(s.readline, ''):

            for i in line:

                count_array[i] = 0

    return count_array


if __name__ == '__main__':
    main()