import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

# param
lda = 3e8 / 10e9
I = 1
l = .1

# mesh_size
n = 100
# plaque
x = np.linspace(-.5, .5, n)
y = np.linspace(-.5, .5, n)
z = - 0.5  # ref : dipole
alpha = 45*np.pi/180


def H_inc(x, y):
    """
    Computes the incident magnetic field radiated onto the reflect array
    :param x: numpy array, x values along the plane
    :param y: numpy array, y values along the plane
    :return: matrix of the magnetic field
    """

    p = np.array([np.sqrt(2) / 2, 0, np.sqrt(2) / 2])        # orientation du dipole
    q = np.array([np.sqrt(2) / 2, 0, -np.sqrt(2) / 2])       # perpendiculaire au dipole

    uz = -0.5 * np.ones(len(x))
    u = np.column_stack((x+.5, y, uz))                       # direction de propagation

    R = norm(u, axis=1)                                      # distance entre plan et dipole
    u_hat = u / R[:, np.newaxis]                             # normalise la direction de propagation

    sintheta = norm(np.cross(u_hat, p), axis=1)              # theta angle entre axe dipole et u
    costhetabis = np.dot(u_hat, q)                           # thetabis angle entre broadside et u

    k = 2 * np.pi / lda                                      # nombre d'onde
    eta = 120*np.pi                                          # impedance du vide

    F_dip = 1j * eta/lda * l/2 * I * sintheta                # radiation pour dipole infinitésimal
    F_horn = costhetabis ** 2                                # radiation de la horn antenna

    E = np.exp(-1j*k*R)/R * F_dip * F_horn
    p_vec = np.tile(p, (len(x), 1))
    dot = np.dot(u_hat, p)[:, np.newaxis]                    # p dot u_hat (commutative)
    E = E[:, np.newaxis] * (p_vec - dot*u_hat)               # Champ électrique émis par la horn

    H = 1/eta * np.cross(u_hat, E)                           # H = 1/eta. û' x E

    return H


def J_ind(x, y):
    """
    Computes the surface current on the reflect array plane
    :param x: numpy array, x values along the plane
    :param y: numpy array, y values along the plane
    :return: matrix of the induced current
    """

    H = H_inc(x, y)             # incident magnetic field
    n = np.array([0, 0, 1])     # normal to the plane
    J = 2 * np.cross(n, H)      # J = 2n x H
    return J


xx, yy = np.meshgrid(x, y)


"""
def F_rad(x, y):
    n = np.array([0, 0, 1])
    u = np.array([x, y, np.sqrt(x ** 2 + y ** 2)])
    u_hat = u / np.linalg.norm(u, axis=1)[:, np.newaxis]
    gt = np.cross(n, gt)
    F = gt - np.dot(gt, u_hat) * u_hat

    return F"""

"""
J = J_ind(x, y)
print(J)
fft = np.fft.fft2(J)
fft = np.fft.fftshift(fft)
plt.pcolormesh(x, y, np.absolute(fft))
plt.colorbar()
plt.show()
"""