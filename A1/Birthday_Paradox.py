
import random
import time
import matplotlib.pyplot as plt


def main():

    domain_size_n = 5000
    trials_k = 0
    m_times = 300
    experiment_results = []

    ##############################################################################
    # Generate numbers in the domain [n] until two have the same value.
    # How many random trials did this take? We will use k to represent this value.

    trials_k = generate_match_in_domain(domain_size_n)
    print(trials_k)

    ##############################################################################
    # Repeat the experiment m = 300 times, and record for each time how many random
    # trials this took. Plot this data as a cumulative density plot where the x-axis
    # records the number of trials required k, and the y-axis records the fraction
    # of experiments that succeeded (a collision) after k trials. The plot should
    # show a curve that starts at a y value of 0, and increases as k increases, and
    # eventually reaches a y value of 1.

    start_time = int(round(time.time() * 1000))
    generate_match_in_domain_mtimes(m_times, domain_size_n, experiment_results)
    timing_in_ms = int(round(time.time() * 1000)) - start_time
    print("timing: " + str(timing_in_ms) + " ms")

    plt.plot(experiment_results)
    plt.title('m = 300')
    plt.ylabel('random trials')
    plt.xlabel('k')
    plt.show()

    ##############################################################################
    # Empirically estimate the expected number of k random trials in order to have
    # a collision. That is, add up all values k, and divide by m.

    sum_of_k_values = 0

    for k in experiment_results:
        sum_of_k_values += k

    estimate = sum_of_k_values / m_times
    print("estimate: " + str(estimate))

    ##############################################################################
    # Describe how you implemented this experiment and how long it took for m = 300
    # trials. Show a plot of the run time as you gradually increase the parameters
    # n and m. (For at least 3 fixed values of between 300 and 10,000, plot the time
    # as a function of n.) You should be able to reach values of n = 1,000,000 and
    # m = 10,000.

    # # m = 300
    # experiment_results_m300 = []
    # experiment_n_m300 = []
    # experiment_timing_m300 = []
    # generate_match_in_domain_mtimes_increasing_to_q(m_times, domain_size_n, experiment_results_m300, 1000000,
    #                                                 experiment_n_m300, experiment_timing_m300)

    #m = 600
    # m_times = 600
    # experiment_results_m600 = []
    # experiment_n_m600 = []
    # experiment_timing_m600 = []
    # generate_match_in_domain_mtimes_increasing_to_q(m_times, domain_size_n, experiment_results_m600, 1000000,
    #                                                 experiment_n_m600, experiment_timing_m600)

    # # m = 1,200
    # m_times = 1200
    # experiment_results_m1200 = []
    # experiment_n_m1200 = []
    # experiment_timing_m1200 = []
    # generate_match_in_domain_mtimes_increasing_to_q(m_times, domain_size_n, experiment_results_m1200, 1000000,
    #                                                 experiment_n_m1200, experiment_timing_m1200)
    #
    # # m = 2,500
    # m_times = 2500
    # experiment_results_m2500 = []
    # experiment_n_m2500 = []
    # experiment_timing_m2500 = []
    # generate_match_in_domain_mtimes_increasing_to_q(m_times, domain_size_n, experiment_results_m2500, 1000000,
    #                                                 experiment_n_m2500, experiment_timing_m2500)

    # m = 5,000
    # m_times = 5000
    # experiment_results_m5000 = []
    # experiment_n_m5000 = []
    # experiment_timing_m5000 = []
    # generate_match_in_domain_mtimes_increasing_to_q(m_times, domain_size_n, experiment_results_m5000, 1000000,
    #                                                 experiment_n_m5000, experiment_timing_m5000)

    # m = 10,000
    # m_times = 10000
    # experiment_results_m10000 = []
    # experiment_n_m10000 = []
    # experiment_timing_m10000 = []
    # generate_match_in_domain_mtimes_increasing_to_q(m_times, domain_size_n, experiment_results_m10000, 1000000,
    #                                                 experiment_n_m10000, experiment_timing_m10000)

    # plt.plot(experiment_n_m300, experiment_timing_m300)
    # plt.title('m = 300')
    # plt.ylabel('time in ms')
    # plt.xlabel('n')
    # plt.show()

    # plt.plot(experiment_n_m600, experiment_timing_m600)
    # plt.title('m = 600')
    # plt.ylabel('time in ms')
    # plt.xlabel('n')
    # plt.show()

    # plt.plot(experiment_n_m1200, experiment_timing_m1200)
    # plt.title('m = 1200')
    # plt.ylabel('time in ms')
    # plt.xlabel('n')
    # plt.show()
    #
    # plt.plot(experiment_n_m2500, experiment_timing_m2500)
    # plt.title('m = 2500')
    # plt.ylabel('time in ms')
    # plt.xlabel('n')
    # plt.show()

    # plt.plot(experiment_n_m5000, experiment_timing_m5000)
    # plt.title('m = 5000')
    # plt.ylabel('time in ms')
    # plt.xlabel('n')
    # plt.show()

    # plt.plot(experiment_n_m10000, experiment_timing_m10000)
    # plt.title('m = 10000')
    # plt.ylabel('time in ms')
    # plt.xlabel('n')
    # plt.show()

    return


##############################################################################
# Generate numbers in the domain [n] until two have the same value.
def generate_match_in_domain(domain):

    trials_k = 0
    match_found = False
    randomly_generated_numbers = {}

    # Loop while match not found
    while match_found is False:

        # Create random number and increment number of trials
        new_random_number = random.randint(0, domain)
        trials_k += 1

        # Check if random number matches any existing numbers
        if new_random_number in randomly_generated_numbers.keys():
            # Mark match found
            match_found = True
            return trials_k
        else:
            # Add the new number to the dictionary and continue
            randomly_generated_numbers[new_random_number] = 0


##############################################################################
# Generate numbers in the domain [n] until two have the same value, m times.
def generate_match_in_domain_mtimes(m, n, experiment_results_trials):

    for experiment in range(0, m):
        trials_k = generate_match_in_domain(n)
        #print(trials_k)
        experiment_results_trials.append(trials_k)

    return


def generate_match_in_domain_mtimes_increasing_to_q(m, n, experiment_results_trials_increasing, q,
                                                    experiment_results_n, experiment_results_time):

    for new_n in range(n, q, 10000):
        start_time = int(round(time.time() * 1000))

        for experiment in range(0, m):
            trials_k = generate_match_in_domain(new_n)
            experiment_results_trials_increasing.append(trials_k)

        timing_in_ms = int(round(time.time() * 1000)) - start_time

        experiment_results_n.append(new_n)
        experiment_results_time.append(timing_in_ms)

        if (new_n % 25000) == 0:
            print("update: " + str(timing_in_ms))

    return


if __name__ == '__main__':
    main()
