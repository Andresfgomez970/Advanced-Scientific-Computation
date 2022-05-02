import numpy as np  # standart data management
import matplotlib.pyplot as plt  # plot routines
from scipy import special  # use for bessel first and second kind


##########################################
# Defining constants for the problems
G = 43007.1  # G in the following units


def Vertical_structure(z, z0):
    """
    Define vertical structure normalized; this permits to arrive to same disk
      potantial withouth changing units for energy
    """
    return (1 / np.cosh(z / z0)) ** 2


def Exponential_disk_potential(R, z, Rs, zs, Sigma0, Npoints, name):
    file = open(name, "w")
    NR, Nz = len(R), len(z)
    # print(NR, Nz)
    result = np.zeros((Nz, NR))
    R_max, z_max = R.max(), z.max()
    for R_c in range(len(R)):
        # print(R_c)
        for z_c in range(len(z)):
            for i in range(Npoints):
                # When at R_max no more considetations for the potential
                #  farther from it.
                a = np.random.random() * R_max * 0.8
                zp = (2 * np.random.random() - 1) * z_max * 0.8

                rp = ((z[z_c] - zp) ** 2 + (a + R[R_c]) ** 2) ** (0.5)
                rn = ((z[z_c] - zp) ** 2 + (a - R[R_c]) ** 2) ** (0.5)
                cte = - 4 * G * Sigma0 / Rs
                ver_s = Vertical_structure(z[z_c], zs)

                result[z_c][R_c] += cte * ver_s *\
                    np.arcsin(2 * a / (rp + rn)) * a *\
                    special.kv(0, a / Rs)

            result[z_c][R_c] = result[z_c][R_c] / Npoints
            file.write(str(z[z_c]) + " " + str(R[R_c]) +
                       " " + str(result[z_c][R_c]) + '\n')

    return result


def Gen_potential_values_zx():
    zs, Rs = 0.3, 2  # Kpc
    # According to Binney
    Sigma0 = 2e-1  # UM/UL

    # Range that constaint 99 percent of the data, only positive since disk
    #    must again contain 99 percent of data
    NR = 50
    R = np.linspace(0, 15.895, NR)  # 15
    # plt.plot(R, np.exp(-R / Rs))
    # plt.show()
    #  be symmetric respect plane relflections.
    Nz = 50
    z = np.linspace(0, 1.002, Nz)  # 1
    # plt.plot(z, Vertical_structure(z, zs))
    # plt.show()
    # Usually 1000 shows the functions as smooth
    Npoints = 1000

    name = "potential_40000.txt"

    Exponential_disk_potential(R, z, Rs, zs, Sigma0, Npoints, name)


def plot_contours():
    ##########################
    # plot xz
    NR = 50
    Nz = 50

    data = np.loadtxt("potential_40000.txt", usecols=(0, 1, 2))
    zv, Rv, result = data[:, 0], data[:, 1], data[:, 2]

    zv.shape = (Nz, NR)
    Rv.shape = (Nz, NR)
    result.shape = (Nz, NR)

    plt.contourf(Rv, zv, result)
    plt.title("Equipotential contours for plane xz")
    plt.xlabel("R")
    plt.ylabel("z")
    plt.savefig("Rz.png")
    plt.show()


Gen_potential_values_zx()
plot_contours()
