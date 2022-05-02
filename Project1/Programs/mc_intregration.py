from abc import ABCMeta, abstractmethod
from scipy.special import erf
from scipy.stats import norm
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mc_mapping import Probability_density

MAX_FLOAT = sys.float_info.max


def gaussian(X, m1, d1):
    C = 1 / (2 * np.pi) ** 0.5 / d1
    return C * np.exp(- (X - m1) ** 2 / (2 * d1 * d1))


class BaseIntFunction(metaclass=ABCMeta):
    @abstractmethod
    def integrate(self, expected_error, N_samples, method):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class IntegrableFunction(object):
    """This routine initialize a function object. Integration cacn be done
        to withon certain error or with a given number of samples."""

    def __init__(self, function, params, limits):
        super(IntegrableFunction, self).__init__()
        ##################################################################
        # General parameters of the function necessary for integration
        ##################################################################
        # Characterize function
        self.function = function
        self.params = params
        self.limits = limits
        self.dim = 1 if isinstance(limits[0], (float, int)) else len(limits[0])

        # Characterize result: variables that are always calculated in the
        #   integral estimation
        self.result = 0
        self.statistic_variance = 0

        # Additional parameters that come when integral is called
        self.method = "standart"
        # Additional variables to evaluate convergence when needed
        self.convergence_confidence = 0.95

        ##################################################################
        # Defined for standart montecarlo integration method and many other
        ##################################################################
        self.N_samples = 100  # default initial value
        self.expected_error = None  # No specified notice
        self.__sample_values = None  # To recognize first time sample

        # Additonal variables for vegas method
        self.__intervals = None
        self.__region_values = None
        self.__pdf_estimate = None
        self.__Es = []
        self.__Ss = []

        # Initialization warning and validation
        self.__init_warnings()
        self.__translate_args()

    def integrate(self, expected_error=None, N_samples=100, method='standart',
                  convergence_confidence=0.64, convergence_type="weak",
                  plot=False):
        # Actualize parameters
        self.expected_error = expected_error
        self.N_samples = N_samples
        self.method = method
        self.convergence_confidence = convergence_confidence

        # In case error is specified we aim for it.
        if isinstance(self.expected_error, (int, float)):
            convergence_state = False
            acutal_error = self.expected_error + 1
            while (convergence_state is False):
                self.__generate_N_sample_points_more()

                # Estimate depending of method
                self.estimate_given_method()

                acutal_error = self.statistic_variance ** 0.5
                if (acutal_error < self.expected_error):
                    # Evaluate integral convergence
                    if (convergence_type == "weak"):
                        convergence_state = self.convergence_weak(True)
                    elif (convergence_type == "medium"):
                        convergence_state = self.convergence_medium()
                    elif (convergence_type == "strong"):
                        convergence_state = self.convergence_strong(plot=plot)

        # In case non expected error is given the integration is carry out
        #  with 100 points.
        elif isinstance(self.N_samples, int):
            self. __generate_N_sample_points()

            # Estimate depending of method
            self.estimate_given_method()

        return self.result, self.statistic_variance ** 0.5, self.N_samples

    def convergence_weak(self, verbose=False):
        E_last = self.result
        self. __generate_N_sample_points()
        self.estimate_given_method()
        self.standart_estimates()

        E_new = self.result
        if verbose:
            diff = np.abs(E_last - E_new)
            print('Npoints: %e, difference: %e' % (self.N_samples, diff))
        if (np.abs(E_last - E_new) < 2 * self.expected_error):
            return True

        return False

    def convergence_medium(self, verbose=False, Nf=100):
        f_means = np.ones(Nf)
        for i in range(len(f_means)):
            f_means[i] = self.result
            self. __generate_N_sample_points()
            self.estimate_given_method()

        mean_result = f_means.mean()
        var_mean_estimate = (f_means - mean_result) ** 2
        var_mean_estimate = var_mean_estimate.sum() / (len(f_means) - 1)

        if (var_mean_estimate ** (0.5) < self.expected_error):
            self.result = mean_result
            self.statistic_variance = var_mean_estimate
            return True

        return False

    def convergence_strong(self, convergence_confidence=0.95, Nf=1000,
                           plot=False):
        """ This routine will take advantage from the fact that mean(f) will
            tend to a gaussian; so a gaussian will be adjusted, and if the
            sigma determined is in correspondence with the one estimated it
            will pass. """

        f_means = np.ones(Nf)
        for i in range(len(f_means)):
            f_means[i] = self.result
            self. __generate_N_sample_points()
            self.estimate_given_method()

        mean_result = f_means.mean()
        var_mean_estimate = (f_means - mean_result) ** 2
        var_mean_estimate = var_mean_estimate.sum() / (len(f_means) - 1)

        f_normalized = (f_means - mean_result) / var_mean_estimate ** 0.5
        n_bins = int(1 + np.log2(Nf))
        p, edges = np.histogram(f_normalized, n_bins, density=True)
        x_random = (edges[1:] + edges[:-1]) / 2

        try:
            popt, pcov = curve_fit(gaussian, x_random, p,
                                   bounds=([-1., 0.5], [1., 2.]))
            print(pcov[0][0] ** 0.5, pcov[1][1] ** 0.5)
        except(ValueError):
            return False

        mean_result = popt[0] * var_mean_estimate ** 0.5 + mean_result
        dev_mean_estimate = popt[1] * var_mean_estimate ** 0.5

        x_cut = - norm.ppf((1 - convergence_confidence) / 2)
        x_cut_real = x_cut * var_mean_estimate ** 0.5
        # The sumation of the mean in the transformation is ommited
        if (x_cut_real < self.expected_error):
            self.result = mean_result
            self.statistic_variance = dev_mean_estimate ** 2

            if(plot):
                # plt.figure(figsize=(10, 8))
                plt.title('Histogram of I estimates')
                width = (x_random.max() - x_random.min()) / n_bins
                plt.bar(x_random, p, align='center', alpha=0.5, width=width,
                        edgecolor="k")
                x_copy = np.linspace(x_random.min(), x_random.max(), 1000)
                label = "theoretical model"
                plt.plot(x_copy, gaussian(x_copy, *popt), label=label)
                plt.xlabel(r'$Z_2$')
                plt.ylabel('p')
                plt.savefig("gaussian_adjust.png")

                plt.show()
            return True

        return False

    def estimate_given_method(self):
        # Estimate depending of method
        if (self.method == 'standart'):
            self.standart_estimates()
        elif (self.method == 'vegas'):
            self.vegas_estimates()

    def standart_estimates(self):
        self.result = self.__sample_values.sum() / self.N_samples

        # Multiply by ranges
        lows, highs = self.limits[0], self.limits[1]

        if self.dim == 1:
            self.result *= (highs - lows)

        else:
            for low, high in zip(lows, highs):
                self.result *= (high - low)

        self.obtain_variance()

    def vegas_estimates(self):
        # will give E and S
        self.result = self.__sample_values.sum() / self.N_samples
        self.__Es.append(self.result)

    def __vegas__init(self):
        # Inital values are obtained
        pass

    def __vegas__state_update(self):
        # Will update all vegas variables
        pass

    def obtain_variance(self):
        mean = self.__sample_values.mean()
        self.statistic_variance = (self.__sample_values - mean) ** 2
        self.statistic_variance = self.statistic_variance.sum()
        self.statistic_variance *= 1 / (self.N_samples - 1)
        # But we are talking about the mean
        self.statistic_variance *= 1 / self.N_samples

        return self.statistic_variance

    def __generate_N_sample_points(self):
        X = np.ones((self.N_samples, self.dim))
        lows, highs = self.limits[0], self.limits[1]
        i_dims = np.arange(self.dim)

        if(self.method == "standart"):
            # Fill X depending on the dimension
            if self.dim == 1:
                X[:, 0] = np.random.uniform(lows, highs, self.N_samples)
            else:
                for low, high, i_dim in zip(lows, highs, i_dims):
                    X[:, i_dim] = np.random.uniform(low, high, self.N_samples)

            self.__sample_values = self.function(X, *self.params)

        if(self.method == "vegas"):
            pass

    def __generate_N_sample_points_more(self):
        X = np.ones((self.N_samples, self.dim))
        lows, highs = self.limits[0], self.limits[1]
        i_dims = np.arange(self.dim)

        if(self.method == "standart"):
            # Fill X depending on the dimension
            if self.dim == 1:
                X[:, 0] = np.random.uniform(lows, highs, self.N_samples)
            else:
                for low, high, i_dim in zip(lows, highs, i_dims):
                    X[:, i_dim] = np.random.uniform(low, high, self.N_samples)

            # When no sample values have been obtained
            if self.__sample_values is None:
                self.__sample_values = self.function(X, *self.params)

            else:
                self.__sample_values = np.append(self.__sample_values,
                                                 self.function(X, *self.params))
                self.N_samples = len(self.__sample_values)

        if (self.method == "vegas"):
            pass

    def __init_warnings(self):
        if np.any(np.array(self.limits) == np.inf):
            m = "Verify that values out of %e and %e in any dimension do " +\
                "not contribute to the function to be integrated; consider" +\
                "changes in variable.\n"
            print(m % (-MAX_FLOAT, MAX_FLOAT))

    def __translate_args(self):
        # Translate limits
        lows, highs = self.limits[0], self.limits[1]
        i_dims = np.arange(self.dim)

        # One dimension check
        if self.dim == 1:
            # Check limits
            if lows == -np.inf:
                self.limits[0] = -MAX_FLOAT

            if highs == np.inf:
                self.limits[1] = MAX_FLOAT
        else:
            for low, high, i_dim in zip(lows, highs, i_dims):
                # Check limits
                if low == -np.inf:
                    self.limits[0][i_dim] = -MAX_FLOAT

                if high == np.inf:
                    self.limits[1][i_dim] = MAX_FLOAT

    def __repr__(self):
        if (self.method == 'standart'):
            E, S = self.result, self.statistic_variance ** 0.5
            N = self.N_samples
            return "E : %f \nS : %f \nN : %d" % (E, S, N)

    def __call__(self, X, *params):
        return self.function(X, *params)


##############################################################################
# Defininig test cases
##############################################################################
def one_dim_tests():
    # Init time of running
    start = time.time()

    # Verify error 0 and perfect result of 1
    def uniform_1d(X, *params):
        dims = X.shape
        return np.ones(dims[0])

    # Verify more reliable result
    def gaussian(X, *params):
        m1, d1 = params
        C = 1 / (2 * np.pi) ** 0.5 / d1
        return C * np.exp(- (X - m1) ** 2 / (2 * d1 * d1))

    print("*" * 100)
    print("Examples with N chosen at hand")
    print("\n\n")
    print("'Perfect' Example: U(0, 1)")
    np.random.seed(1)
    uniform_object = IntegrableFunction(uniform_1d, (0, ), [0, 1])
    uniform_object.integrate()
    print(uniform_object)

    print("\n\n")
    print("Sucess Example: N(0, 1) from 0 to 10")
    np.random.seed(1)
    gaussian_object = IntegrableFunction(gaussian, (0, 1), [0, 10])
    E, S, N = gaussian_object.integrate()
    print(gaussian_object)
    print("relative error:", abs(E - erf(10 / 2) / 2))

    print("\n\n")
    print("Ill Example: N(0, 1) from 0 to 100")
    np.random.seed(1)
    gaussian_object = IntegrableFunction(gaussian, (0, 1), [0, 100])
    E, S, N = gaussian_object.integrate()
    print(gaussian_object)
    print("relative error:", abs(E - erf(1000 / 2) / 2))
    print("\n\n")

    print("Ill Example Fixed: N(0, 1) from 0 to 100")
    np.random.seed(1)
    gaussian_object = IntegrableFunction(gaussian, (0, 1), [0, 100])
    E, S, N = gaussian_object.integrate(N_samples=10000)
    print(gaussian_object)
    print("relative error:", abs(E - erf(1000 / 2) / 2))
    print("*" * 50 + "\n")

    print("*" * 100)
    print("Dynamic integration for N(0, 1) from 0 to 100")
    print("\n\n")
    print("Expected error of 0.05")
    np.random.seed(1)
    gaussian_object = IntegrableFunction(gaussian, (0, 1), [0, 100])
    E, S, N = gaussian_object.integrate(expected_error=0.05)
    print(gaussian_object)
    print("relative error:", abs(E - erf(1000 / 2) / 2))

    print("\n\n")
    print("Expected error of 0.01")
    np.random.seed(1)
    gaussian_object = IntegrableFunction(gaussian, (0, 1), [0, 100])
    E, S, N = gaussian_object.integrate(expected_error=0.01)
    print(gaussian_object)
    print("relative error:", abs(E - erf(100 / 2) / 2))

    end = time.time()
    print("It took %f s" % (end - start))


def two_dim_test():
    # Init time of running
    start = time.time()

    # Define to obtain perfect result in the calculation
    def z1_plane(X, *params):
        dims = X.shape
        return np.ones((dims[0], 1))

    print("*" * 100)
    np.random.seed(1)
    print("'Perfect' Example: (X, Y) ~ U(0, 1)")
    z1_plane_object = IntegrableFunction(z1_plane, (0, ), ((0, 0), (1, 1)))
    z1_plane_object.integrate()
    print(z1_plane_object)

    def independent_gaussian_2d(X, *params):
        m1, m2, d1, d2 = params
        C = 1 / (2 * np.pi * d1 * d2)
        mu, sigmas = np.array([m1, m2]), np.array([d1 * d1, d2 * d2])
        result = np.exp(-(X - mu) ** 2 / (2 * sigmas))
        return C * result[:, 0] * result[:, 1]

    print("\n\n")
    print("Dynamic integration for N([0,0,1,1]) from 0, 0 to 10, 10")
    print("Expected error of 0.05, weak convergence")
    np.random.seed(int(time.time()))
    gaussian_object = IntegrableFunction(independent_gaussian_2d, (0, 0, 1, 1),
                                         [[0, 0], [10, 10]])
    E, S, N = gaussian_object.integrate(expected_error=0.04)
    print(gaussian_object)
    print("relative error:", abs(E - (erf(10 / 2) / 2) ** 2))

    print("\n\n")
    print("Dynamic integration for N([0,0,1,1]) from 0, 0 to 10, 10")
    print("Expected error of 0.05, medium convergence")
    np.random.seed(int(time.time()))
    gaussian_object = IntegrableFunction(independent_gaussian_2d, (0, 0, 1, 1),
                                         [[0, 0], [10, 10]])
    E, S, N = gaussian_object.integrate(expected_error=0.04,
                                        convergence_type='medium',
                                        N_samples=100)
    print(gaussian_object)
    print("relative error:", abs(E - (erf(10 / 2) / 2) ** 2))

    print("\n\n")
    print("Dynamic integration for N([0,0,1,1]) from 0, 0 to 10, 10")
    print("Expected error of 0.05, strong convergence")
    np.random.seed(int(time.time()))
    gaussian_object = IntegrableFunction(independent_gaussian_2d, (0, 0, 1, 1),
                                         [[0, 0], [10, 10]])
    E, S, N = gaussian_object.integrate(expected_error=0.04,
                                        convergence_type='strong',
                                        N_samples=100,
                                        plot=True)
    print(gaussian_object)
    print("relative error:", abs(E - (erf(10 / 2) / 2) ** 2))

    # Close runing time
    print("\n\n")
    end = time.time()
    print("It took %f s" % (end - start))


def vegas_example():
    pass


def main():
    # TO DO: redefine lines of code of the examples
    # TO DO: evaluate a confidence interval in the strong approximation
    # TO DO: make a distribution of the relative error. Should it be related
    #  to the f_means distribution directly?
    # TO DO: generalize for n-dimensions the Probability_density class

    one_dim_tests()
    two_dim_test()


if __name__ == "__main__":
    main()
