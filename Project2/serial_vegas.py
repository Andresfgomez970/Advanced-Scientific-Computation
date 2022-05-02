import numpy as np
import time
# import matplotlib.pyplot as plt
# from array import array


def gaussian(x):
    return 1. / (2 * np.pi) ** 0.5 / s * np.exp(- x ** 2 / 2)


def rejection(p, M):
    mapped_values = np.ones(M)
    p_mapped_values = np.ones(M)
    n_mapped_values = 0
    while (n_mapped_values != M):
        x = val_0[0] + np.random.rand() * (val_f[0] - val_0[0])
        y = np.random.rand() * val_f[1]
        con1 = edges < x
        index = np.where(edges == edges[con1][-1])[0][0]
        if y < p[index]:
            mapped_values[n_mapped_values] = x
            p_mapped_values[n_mapped_values] = p[index]
            n_mapped_values += 1
    return mapped_values, gaussian(mapped_values), p_mapped_values


def estimates(values, p_mapped_values):
    I = (values / p_mapped_values)
    I = I.sum() / len(values)
    var = 1. / (len(values) * (len(values) - 1)) * \
        ((values ** 2 / p_mapped_values).sum() - I**2)
    sigma = var ** (0.5)
    return I, sigma


def update_state(it, mapped_values, values):
    # Calulate mi's
    for i in range(1, len(edges)):
        con1 = mapped_values / edges[i] <= 1
        values_in_each_interval[f"interval{i}"] = values[con1]
        mapped_values = mapped_values[~con1]
        values = values[~con1]

    # Correct possible division by 0
    sums = np.zeros(len(edges) - 1)
    for i in range(1, len(edges)):
        if len(values_in_each_interval[f"interval{i}"]) != 0:
            sums[i - 1] = values_in_each_interval[f"interval{i}"].sum() / len(values_in_each_interval[f"interval{i}"])

    mi = K * (edges[1:] - edges[:-1]) * sums / Is[it - 1]

    # Actualize edges and p
    cumsum_mi = np.cumsum(mi)
    intervals_per_range = cumsum_mi[-1] / (len(edges) - 1)
    mi_copy = mi.copy()
    intervals_size_mi = np.divide(edges[1:] - edges[:-1], mi,
                                  out=np.zeros_like(mi), where=mi != 0)

    x_sum = 0
    total_mi = 0

    for i in range(len(edges) - 2):
        cumsum_mi = cumsum_mi - intervals_per_range
        con_sum = cumsum_mi < 0
        mi_intervals_sum = 0

        for j in range(len(mi_copy[con_sum])):
            x_sum += intervals_size_mi[j] * \
                (cumsum_mi[con_sum][j] + intervals_per_range)
            mi_intervals_sum += (cumsum_mi[con_sum][j] + intervals_per_range)
            total_mi += (cumsum_mi[con_sum][j] + intervals_per_range)

        if (mi_intervals_sum < intervals_per_range and (len(mi_copy[con_sum]) != len(intervals_size_mi))):
            x_sum += (intervals_per_range - mi_intervals_sum) * \
                intervals_size_mi[len(mi_copy[con_sum])]
            mi_intervals_sum += (intervals_per_range - mi_intervals_sum)
            total_mi += (intervals_per_range - mi_intervals_sum)
            mi_copy[len(mi_copy[con_sum])] = mi_copy[len(mi_copy[con_sum])] - \
              (intervals_per_range - mi_intervals_sum)

        edges[i + 1] = x_sum
        # Actualize next round
        mi_copy = mi_copy[~con_sum]
        cumsum_mi = cumsum_mi[~con_sum]
        intervals_size_mi = intervals_size_mi[~con_sum]

    p = 1 / (N * (edges[1:] - edges[:-1]))
    return p, mi, edges


if __name__ == "__main__":
    start = time.time()  # Line to count general timing of program
    # general_times = open("data/general_time.txt", "a")
    # Global variables declared for 1dim example
    # estimate_times = open("data/estimates_time.txt", "a")
    # adapt_times = open("data/adapt_time.txt", "a")
    rej_times = open("data/rejection_timetest.txt", "a")
    # convergence_times = open("data/convergence_time.txt", "a")

    f = open("gaussian_test1dim.txt", "w")

    N = 100
    M = N * 100
    K = 1000
    val_0 = [0, 0]
    val_f = [4, 1.5]

    edges = np.linspace(val_0[0], val_f[0], N + 1)
    p = 1 / (N * (edges[1:] - edges[:-1]))

    values_in_each_interval = {}
    Is, Sigmas = [], []
    max_iterations = 100
    iteration = 1
    m = 0
    s = 1

    est_total = 0
    adp_total = 0
    rej_total = 0
    con_total = 0
    mapped_values, values, p_mapped_values = rejection(p, M)
    while (iteration <= max_iterations):
        # est_start = time.time()
        Iit, sigmait = estimates(values, p_mapped_values)
        Is.append(Iit)
        Sigmas.append(sigmait)
        # print(Iit, sigmait)
        f.write(f"{Iit, sigmait}\n")
        # est_end = time.time()
        # est_total += (est_end - est_start)

        # adp_start = time.time()
        p, mi, edges = update_state(iteration, mapped_values, values)
        # adp_end = time.time()
        # adp_total += (adp_end - adp_start)

        rej_start = time.time()
        mapped_values, values, p_mapped_values = rejection(p, M)
        rej_end = time.time()
        rej_total += (rej_end - rej_start)

        iteration += 1

        # con_start = time.time()
        convergence_state = np.all(np.abs(mi - mi[0]) < 1)
        # con_end = time.time()
        # con_total += (con_end - con_start)

        if (convergence_state):
            break

    f.close()
    ###################
    # Reserved comment for general program count time
    end = time.time()
    # general_times.write(f"{end - start}\n")
    # general_times.close()
    #######
    # estimate_times.write(f"{est_total}\n")
    # adapt_times.write(f"{adp_total}\n")
    rej_times.write(f"{rej_total}\n")
    # print(rej_total)
    # convergence_times.write(f"{con_total}\n")
    # estimate_times.close()
    # adapt_times.close()
    rej_times.close()
    # convergence_times.close()
# End
