import numpy as np
import time
import multiprocessing as mp
import itertools


def gaussian(x, m, s):
    return 1. / (2 * np.pi) ** 0.5 / s * np.exp(- (x - m) ** 2 / (2 * s * s))


class Basic_vegas(object):
    """docstring for Probability_density"""

    def __init__(self):
        super(Basic_vegas, self).__init__()

        self.N = 100
        self.M = self.N * 100
        self.K = 1000
        self.val_0 = [0, 0]
        self.val_f = [4, 1.5]

        self.mapped_values = np.ones(self.M)
        self.p_mapped_values = np.ones(self.M)
        self.values = np.ones(self.M)
        self.edges = np.linspace(self.val_0[0], self.val_f[0], self.N + 1)
        self.p = 1 / (self.N * (self.edges[1:] - self.edges[:-1]))
        self.n_mapped_values = 0

        self.values_in_each_interval = {}
        self.Ins, self.Sigmas = [], []
        self.max_iterations = 2
        self.iteration = 0
        self.m = 0
        self.s = 1

        self.pdf = gaussian

        self.pool = 0

        self.convergence_state = 0

        self.init_parallel()

    def init_parallel(self):
        self.N_processors = mp.cpu_count()
        # mp.set_start_method('fork')
        self.pool = mp.Pool(self.N_processors)

    def rejection(self, M):
        self.M = M
        while (self.n_mapped_values < self.M):
            x = self.val_0[0] + np.random.rand() * \
                (self.val_f[0] - self.val_0[0])
            y = np.random.rand() * self.val_f[1]
            con1 = self.edges < x
            index = np.where(self.edges == self.edges[con1][-1])[0][0]
            if y < self.p[index]:
                self.mapped_values[self.n_mapped_values] = x
                self.p_mapped_values[self.n_mapped_values] = self.p[index]
                self.values[self.n_mapped_values] = self.pdf(x, self.m, self.s)
                self.n_mapped_values += 1

    def paralelized_rejection(self):
        chuck_size = 500  # self.M // self.N_processors * 4
        remanent = self.M % chuck_size
        N_ranges = int(self.M / chuck_size)
        Regions = np.ones(N_ranges) * chuck_size
        Regions[0] += remanent

        self.n_mapped_values = 0
        np.random.seed()
        # only map give worse results.
        self.pool.map_async(self.rejection, Regions)

    def estimates(self):
        In = (self.values / self.p_mapped_values)
        In = In.sum() / len(self.values)
        var = 1. / (len(self.values) * (len(self.values) - 1)) * \
            ((self.values ** 2 / self.p_mapped_values).sum() - In**2)
        sigma = var ** (0.5)
        self.Ins.append(In)
        self.Sigmas.append(sigma)

    def update_state(self):
        #################################
        # Calulate mi's
        mapped_values_copy = self.mapped_values.copy()
        values_copy = self.values.copy()
        for i in range(1, len(self.edges)):
            con1 = mapped_values_copy / self.edges[i] <= 1
            self.values_in_each_interval[i] = values_copy[con1]
            mapped_values_copy = mapped_values_copy[~con1]
            values_copy = values_copy[~con1]

        # It will save the integrate in each regiondivided by its interval
        sums = np.zeros(len(self.edges) - 1)
        for i in range(1, len(self.edges)):
            if len(self.values_in_each_interval[i]) != 0:
                sums[i - 1] = self.values_in_each_interval[i].sum() / \
                    len(self.values_in_each_interval[i])

        mi = self.K * (self.edges[1:] - self.edges[:-1]) * sums / \
            self.Ins[self.iteration - 1]

        # Actualize edges and p
        cumsum_mi = np.cumsum(mi)
        intervals_per_range = cumsum_mi[-1] / (len(self.edges) - 1)
        mi_copy = mi.copy()
        intervals_size_mi = np.divide(self.edges[1:] - self.edges[:-1], mi,
                                      out=np.zeros_like(mi), where=mi != 0)

        # It will save the succesive intervals sum to permit us to define the
        #   new intervals Delta{x_i}.
        x_sum = 0
        total_mi = 0

        for i in range(len(self.edges) - 2):
            cumsum_mi = cumsum_mi - intervals_per_range
            con_sum = cumsum_mi < 0
            mi_intervals_sum = 0

            for j in range(len(mi_copy[con_sum])):
                j_intervals = cumsum_mi[con_sum][j] + intervals_per_range
                x_sum += intervals_size_mi[j] * j_intervals
                mi_intervals_sum += cumsum_mi[con_sum][j] + intervals_per_range
                total_mi += j_intervals

            if (mi_intervals_sum < intervals_per_range
                    and (len(mi_copy[con_sum]) != len(intervals_size_mi))):
                remanent_intevals = (intervals_per_range - mi_intervals_sum)
                x_sum += remanent_intevals * \
                    intervals_size_mi[len(mi_copy[con_sum])]
                mi_intervals_sum += remanent_intevals
                total_mi += remanent_intevals
                mi_copy[len(mi_copy[con_sum])] -= remanent_intevals

            self.edges[i + 1] = x_sum
            # Actualize next round
            mi_copy = mi_copy[~con_sum]
            cumsum_mi = cumsum_mi[~con_sum]
            intervals_size_mi = intervals_size_mi[~con_sum]

        self.convergence_state = np.all(np.abs(mi - mi[0]) < 1)
        self.p = 1 / (self.N * (self.edges[1:] - self.edges[:-1]))

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict


def serial_test():
    start = time.time()
    basic_vegas = Basic_vegas()
    basic_vegas.rejection(basic_vegas.M)
    while (basic_vegas.iteration < basic_vegas.max_iterations
           and ~basic_vegas.convergence_state):
        # basic_vegas.estimates()
        basic_vegas.n_mapped_values = 0
        basic_vegas.rejection(basic_vegas.M)
        basic_vegas.iteration += 1
        # basic_vegas.update_state()
    basic_vegas.pool.close()
    basic_vegas.pool.join()
    end = time.time()
    print(end - start)


def parallel_test():
    start = time.time()
    basic_vegas_p = Basic_vegas()
    basic_vegas_p.paralelized_rejection()
    while ((basic_vegas_p.iteration < basic_vegas_p.max_iterations)
           and ~basic_vegas_p.convergence_state):
        basic_vegas_p.estimates()
        basic_vegas_p.paralelized_rejection()
        basic_vegas_p.iteration += 1
        basic_vegas_p.update_state()
    basic_vegas_p.pool.close()
    basic_vegas_p.pool.join()
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    serial_test()
    parallel_test()
