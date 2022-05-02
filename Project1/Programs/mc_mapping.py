import random
import math as m
import numpy as np
import matplotlib.pyplot as plt


class Probability_density(object):
    """docstring for Probability_density"""

    def __init__(self, pdf, params, limits):
        super(Probability_density, self).__init__()
        # In general self.pdf can be a single pdf or a list of them, in
        #  the case it is a list the rest of parameters are too.
        self.pdf = pdf
        self.params = params
        self.limits = limits
        self.N_samples = 1000
        # each row is a sample value
        self.__sample_values = []

    def rejection(self, N_samples=1000):
        """
        limits for rejection also include the values min and max value in y.
        """
        self.N_samples = N_samples

        N_samples_colected = False
        while (N_samples_colected is False):
            x_propose = np.random.uniform(self.limits[0][0], self.limits[1][0],
                                          self.N_samples)
            y_propose = np.random.uniform(self.limits[0][1], self.limits[1][1],
                                          self.N_samples)
            yes_condition = y_propose < self.pdf(x_propose, *self.params)
            x_accept = x_propose[yes_condition]

            # Init the first time
            if len(self.__sample_values) == 0:
                self.__sample_values = x_accept
            # add when has been added, st lacking points remain or all is
            #   done.
            else:
                if len(self.__sample_values) == self.N_samples:
                    N_samples_colected = True

                lack_number = self.N_samples - len(self.__sample_values)
                if len(x_propose[yes_condition]) <= lack_number:
                    self.__sample_values = np.append(self.__sample_values,
                                                     x_accept)

                else:
                    self.__sample_values = np.append(self.__sample_values,
                                                     x_propose[:lack_number])

        return self.__sample_values

    def __generate_N_sample_points(self):
        X = np.ones((self.N_samples, self.dim))
        lows, highs = self.limits[0], self.limits[1]
        i_dims = np.arange(self.dim)

        # Fill X depending on the dimension
        if self.dim == 1:
            X[:, 0] = np.random.uniform(lows, highs, self.N_samples)
        else:
            for low, high, i_dim in zip(lows, highs, i_dims):
                X[:, i_dim] = np.random.uniform(low, high, self.N_samples)

        self.__sample_values = self.function(X, *self.params)

        pass

    def metropolis(self, N_samples=1000):
        if self.N_samples is None:
            self.N_samples = N_samples

        # i-th sample value
        xi = self.__gen_xi()
        i = 1
        self.__sample_values.append(xi)
        while i < self.N_samples:
            # proposed new value , note that xi = self.__gen_xi() will
            #   be highly inneficient since points once point is chosen
            #   in near the middle the probability of transitions outside
            #   the middle are low.
            xp = self.__gen_xp(xi)
            # probability ratio
            ratio = self.pdf(xp, *self.params) / self.pdf(xi, *self.params)
            if -m.log(ratio) < 0:
                self.__sample_values.append(xp)
                xi = xp
                i += 1
            else:
                r = random.random()
                if r < ratio:
                    self.__sample_values.append(xp)
                    xi = xp
                    i += 1

        return self.__sample_values

    def __gen_xi(self):
        interval = (self.limits[1] - self.limits[0])
        return self.limits[0] + random.random() * interval

    def __gen_xp(self, xi):
        # Note that uniform cannot be used here since it will not permit
        #  movement from internal parts to external, so movement from an
        #  arbitrary state to other is not satisfied.
        return np.random.normal(xi, 0.5, 1)[0]


def metropolis_example(Gaussian):
    print('\n\n')
    print("Metropolis Example")
    random.seed(1)
    GaussianObject = Probability_density(Gaussian, (0., 1.), (-4., 4.))
    GaussianObject.N_samples = 1e5
    sample_values = GaussianObject.metropolis()

    n_bins = int(GaussianObject.N_samples ** (1 / 3.))
    p, edges = np.histogram(sample_values, n_bins, density=True)

    x_random = (edges[1:] + edges[:-1]) / 2
    x_con = np.linspace(x_random.min(), x_random.max(), 100)
    bar_width = (edges[1] - edges[0]) * 1
    plt.figure(figsize=(10, 10))
    plt.bar(x_random, p, align='center', alpha=0.5, width=bar_width,
            edgecolor="k")
    plt.plot(x_con, Gaussian(x_con, 0, 1))
    plt.show()


def rejection_example(Gaussian):
    print('\n\n')
    print("Rejection Example")
    random.seed(1)
    GaussianObject = Probability_density(Gaussian, (0., 1.),
                                         [[-5., 0], [5., 0.4]])
    sample_values = GaussianObject.rejection()

    n_bins = int(1 + np.log2(GaussianObject.N_samples))
    p, edges = np.histogram(sample_values, n_bins, density=True)

    x_random = (edges[1:] + edges[:-1]) / 2
    x_con = np.linspace(x_random.min(), x_random.max(), 100)
    bar_width = (edges[1] - edges[0]) * 1
    plt.figure(figsize=(10, 10))
    plt.bar(x_random, p, align='center', alpha=0.5, width=bar_width,
            edgecolor="k")
    plt.plot(x_con, Gaussian(x_con, 0, 1))
    plt.show()


def One_dim_example():
    def Gaussian(x, mean, std):
        C = 1 / ((2 * m.pi) ** (0.5) * std)
        return C * np.exp(-(x - mean) ** 2 / (2 * std * std))

    metropolis_example(Gaussian)
    rejection_example(Gaussian)


def main():
    One_dim_example()


if __name__ == "__main__":
    main()
