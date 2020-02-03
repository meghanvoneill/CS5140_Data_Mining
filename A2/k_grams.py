import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np

# Meghan V. O'Neill
# A2 - Document Similarity and Hashing
# CS 5140 Data Mining, Spring 2020
# Professor Phillips


_memomask = {}


def main():
    file_string_D1 = "D1.txt"
    file_string_D2 = "D2.txt"
    file_string_D3 = "D3.txt"
    file_string_D4 = "D4.txt"

    # G1
    G1_k_grams_D1 = k_gram_chars(file_string_D1, 2)
    print("G1 D1: " + str(len(G1_k_grams_D1)))

    G1_k_grams_D2 = k_gram_chars(file_string_D2, 2)
    print("G1 D2: " + str(len(G1_k_grams_D2)))

    G1_k_grams_D3 = k_gram_chars(file_string_D3, 2)
    print("G1 D3: " + str(len(G1_k_grams_D3)))

    G1_k_grams_D4 = k_gram_chars(file_string_D4, 2)
    print("G1 D4: " + str(len(G1_k_grams_D4)))

    # G2
    G2_k_grams_D1 = k_gram_chars(file_string_D1, 3)
    print("G2 D1: " + str(len(G2_k_grams_D1)))

    G2_k_grams_D2 = k_gram_chars(file_string_D2, 3)
    print("G2 D1: " + str(len(G2_k_grams_D2)))

    G2_k_grams_D3 = k_gram_chars(file_string_D3, 3)
    print("G2 D3: " + str(len(G2_k_grams_D3)))

    G2_k_grams_D4 = k_gram_chars(file_string_D4, 3)
    print("G2 D4: " + str(len(G2_k_grams_D4)))

    # G3
    G3_k_grams_D1 = k_gram_strings(file_string_D1, 2)
    print("G3 D1: " + str(len(G3_k_grams_D1)))

    G3_k_grams_D2 = k_gram_strings(file_string_D2, 2)
    print("G3 D2: " + str(len(G3_k_grams_D2)))

    G3_k_grams_D3 = k_gram_strings(file_string_D3, 2)
    print("G3 D3: " + str(len(G3_k_grams_D3)))

    G3_k_grams_D4 = k_gram_strings(file_string_D4, 2)
    print("G3 D4: " + str(len(G3_k_grams_D4)))

    JS = jaccard_similarity(G1_k_grams_D1, G1_k_grams_D2)
    print("G1 JS D1 D2: " + str(JS))

    JS = jaccard_similarity(G1_k_grams_D1, G1_k_grams_D3)
    print("G1 JS D1 D3: " + str(JS))

    JS = jaccard_similarity(G1_k_grams_D1, G1_k_grams_D4)
    print("G1 JS D1 D4: " + str(JS))

    JS = jaccard_similarity(G1_k_grams_D2, G1_k_grams_D3)
    print("G1 JS D2 D3: " + str(JS))

    JS = jaccard_similarity(G1_k_grams_D2, G1_k_grams_D4)
    print("G1 JS D2 D4: " + str(JS))

    JS = jaccard_similarity(G1_k_grams_D3, G1_k_grams_D4)
    print("G1 JS D3 D4: " + str(JS))

    JS = jaccard_similarity(G2_k_grams_D1, G2_k_grams_D2)
    print("G2 JS D1 D2: " + str(JS))

    JS = jaccard_similarity(G2_k_grams_D1, G2_k_grams_D3)
    print("G2 JS D1 D3: " + str(JS))

    JS = jaccard_similarity(G2_k_grams_D1, G2_k_grams_D4)
    print("G2 JS D1 D4: " + str(JS))

    JS = jaccard_similarity(G2_k_grams_D2, G2_k_grams_D3)
    print("G2 JS D2 D3: " + str(JS))

    JS = jaccard_similarity(G2_k_grams_D2, G2_k_grams_D4)
    print("G2 JS D2 D4: " + str(JS))

    JS = jaccard_similarity(G2_k_grams_D3, G2_k_grams_D4)
    print("G2 JS D3 D4: " + str(JS))

    JS = jaccard_similarity(G3_k_grams_D1, G3_k_grams_D2)
    print("G3 JS D1 D2: " + str(JS))

    JS = jaccard_similarity(G3_k_grams_D1, G3_k_grams_D3)
    print("G3 JS D1 D3: " + str(JS))

    JS = jaccard_similarity(G3_k_grams_D1, G3_k_grams_D4)
    print("G3 JS D1 D4: " + str(JS))

    JS = jaccard_similarity(G3_k_grams_D2, G3_k_grams_D3)
    print("G3 JS D2 D3: " + str(JS))

    JS = jaccard_similarity(G3_k_grams_D2, G3_k_grams_D4)
    print("G3 JS D2 D4: " + str(JS))

    JS = jaccard_similarity(G3_k_grams_D3, G3_k_grams_D4)
    print("G3 JS D3 D4: " + str(JS))

    ###################################################################
    # Using grams G2, build a min-hash signature for document D1 and D2
    #   using t = {20, 60, 150, 300, 600} hash functions.

    # Turn set into vector.

    # Create entire set.
    # D1_and_D2_set = G2_k_grams_D1.union(G2_k_grams_D2)
    # D1_and_D2_dict = {}
    # index = 0
    #
    # for k_gram in D1_and_D2_set:
    #     D1_and_D2_dict[k_gram] = index
    #     index += 1
    #
    # D1_vector = np.zeros((1, len(D1_and_D2_set)))
    # D2_vector = np.zeros((1, len(D1_and_D2_set)))
    #
    # for k_gram in G2_k_grams_D1:
    #     index = D1_and_D2_dict[k_gram]
    #     D1_vector[0][index] = 1
    # print(D1_vector)
    #
    # for k_gram in G2_k_grams_D2:
    #     index = D1_and_D2_dict[k_gram]
    #     D2_vector[0][index] = 1
    # print(D2_vector)

    # Use t hash functions to form the family of hash functions.
    t = 600
    D1_grams_list = list(G2_k_grams_D1)
    D2_grams_list = list(G2_k_grams_D2)
    min_vector = []

    # Loop through hash functions.
    for h in range(0, t):
        D1_vector = []
        D2_vector = []

        # Find all the hashed values for D1's k-grams.
        for a in D1_grams_list:
            new_val = hash_function(a, h)
            D1_vector.append(new_val)

        # Find all the hashed values for D2's k-grams.
        for b in D2_grams_list:
            #print("b: " + b)
            new_val = hash_function(b, h)
            D2_vector.append(new_val)

        # Find the minimum for both sets and find the Jaccard similarity.
        min_a = min(D1_vector)
        min_b = min(D2_vector)

        print(jaccard_similarity(set(D1_vector), set(D2_vector)))

        # Store a 1 if a = b, and a 0 otherwise.
        print(min_a)
        print(min_b)
        if min_a == min_b:
            min_vector.append(1)
        else:
            min_vector.append(0)

    print("min_vector: " + str(min_vector))

    partial_sum = 0
    for val in min_vector:
        partial_sum += val

    print(partial_sum)
    js = float(float(partial_sum) / t)
    print("JS: " + str(js))

    # # Loop through the k-grams for D1 of G2.
    # for i in range(0, len(D1_grams_list)):
    #
    #     a = D1_grams_list[i]
    #     new_min_of_a_hash = math.inf
    #
    #     # Use the hash family of functions to get every hash for this k-gram, a.
    #     for hash_function_index in range(0, t):
    #
    #         new_hash = hash_function(a, hash_function_index)
    #
    #         # If the new hash is smaller than the min hash found, update the min hash and store it.
    #         if new_hash < new_min_of_a_hash:
    #             new_min_of_a_hash = new_hash
    #             D1_vector[i] = new_hash
    #
    # # Loop through the k-grams for D2 of G2.
    # for i in range(0, len(D2_grams_list)):
    #
    #     b = D2_grams_list[i]
    #     new_min_of_b_hash = math.inf
    #
    #     # Use the hash family of functions to get every hash for this k-gram, b.
    #     for hash_function_index in range(0, t):
    #
    #         new_hash = hash_function(b, hash_function_index)
    #
    #         # If the new hash is smaller than the min hash found, update the min hash and store it.
    #         if new_hash < new_min_of_b_hash:
    #             new_min_of_b_hash = new_hash
    #             D2_vector[i] = new_hash

    # print(D1_vector)
    # print(D2_vector)
    #
    # js = cumulative_jaccard_similarity(D1_vector, D2_vector, t)
    # print("JS: " + str(js))

    return


def k_gram_chars(file_name, k):
    
    k_grams = []
    f = open(file_name, "r")

    if f.mode == 'r':
        contents = f.read()
    else:
        return

    for index in range(0, len(contents) - k + 1):
        new_k_gram = contents[index:index + k]
        k_grams.append(new_k_gram)

    return set(k_grams)


def k_gram_strings(file_name, k):

    k_grams = []
    f = open(file_name, "r")

    if f.mode == 'r':
        contents = f.read()
    else:
        return

    contents_string_list = contents.split(" ")
    contents_string_list = list(filter(None, contents_string_list))

    for index in range(0, len(contents_string_list) - k + 1):
        new_k_gram = " ".join(contents_string_list[index:index + k])
        k_grams.append(new_k_gram)

    return set(k_grams)


# The Jaccard similarity between A and B is:
#   |(A n B)| / |(A u B)|
def jaccard_similarity(set_A, set_B):

    AnB = set_A.intersection(set_B)
    AuB = set_A.union(set_B)

    j_similarity = (len(AnB) / len(AuB))

    return j_similarity


def cumulative_jaccard_similarity(setA, setB, t):

    summation = 0

    for i in range(min(len(setA), len(setB))):
        # If a and b exist, check whether they have the same value.
        if i < len(setA) and i < len(setB):
            print("a: " + str(setA[i]) + " b: " + str(setB[i]))
            if setA[i] == setB[i]:
                summation += 1

    j_similarity = float(float(1 / t) * float(summation))
    return j_similarity


def min_hash(vectors, t):

    min_vectors = []

    # 3. Repeat steps 1 & 2, t times.
    for step in range(0, t):

        # 1. Randomly reorder (permute) the rows.
        zipped = zip(vectors)
        random.shuffle(zipped)

        # 2. For each set / column, find the top / first 1 bit.
        for vector in zipped:

            # Hash here
            min_vector = []

            # Grab the lowest index number key
            for index in range(0, len(vector)):

                if vector[index] == 1:
                    min_index = index
                    min_vector.append(min_index)
                    break

        min_vectors.append(min_vector)

    return min_vectors


# Code on hash function family from:
#   https://stackoverflow.com/questions/2255604/hash-functions-family-generator-in-python
# Credit to Alex Martelli.
def hash_function(x, n):

    mask = _memomask.get(n)

    if mask is None:
        random.seed(n)
        mask = _memomask[n] = random.getrandbits(32)

    m = 100000
    val = (hash(x) ^ mask) % m
    return val


if __name__ == '__main__':
    main()
