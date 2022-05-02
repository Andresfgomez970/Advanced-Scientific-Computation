import matplotlib.pyplot as plt
import numpy as np


def plot_cv(systems):
    """
    This routine plot the heat capacities as a function of temperature for
      each system from the Energies moments obtained.
    """
    # Intended lables known beforehand
    labels = [2, 4, 6, 8, 16, 32, 64]

    plt.title("Cv vs T from Dissipation Theorem")
    for i in range(len(systems)):
        data = np.genfromtxt(systems[i])
        T, E, E2 = data[1:, 0], data[1:, 1], data[1:, 2]
        cv = 1 / (T ** 2 * labels[i] * labels[i]) * (E2 - E ** 2)
        plt.plot(T, cv, '--', label=labels[i])

    # General labels
    plt.xlabel(r"$T$")
    plt.ylabel(r"$c_v$")
    plt.xlim(T.min(), T.max())
    plt.legend()
    plt.savefig("periodic2.png")
    plt.show()


#################################
# Graphs of Cv ##################
#################################
# Making list of periodic systems
systems = np.genfromtxt("periodic_names640Mp50.txt", dtype="str")
plot_cv(systems[:])
