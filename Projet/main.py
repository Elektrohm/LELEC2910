import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.fft import fft, ifft

# param
lda = 3e8 / 10e9
I = 1
l = 0.1

# mesh_size
n = 100
# meshgrid
x = np.linspace(0, 1, n)
y = np.linspace(-.5, .5, n)
z = - 0.5  # ref : dipole
x, y = np.meshgrid(x, y)


def J(x, y):
    # cartesian to spherical

    xb = np.sqrt(2) / 2 * (x + -z)
    yb = y
    zb = np.sqrt(2) / 2 * (x + z)

    phi = np.arctan2(yb, xb)
    theta = np.arctan2(np.sqrt(xb ** 2 + yb ** 2), zb)
    theta_bis = np.pi / 2 - theta

    k = 2 * np.pi / lda
    R = 0.5 / np.cos(theta_bis + np.pi / 4) * 1 / np.cos(phi)

    J = 1j * I*l / lda * np.sin(theta) * np.sinc(k * l / 2 * np.cos(theta)) * np.exp(-1j * k * R) / R * np.cos(
        theta_bis) ** 2

    J = J * np.sqrt(np.sin(phi) ** 2 * 1 / 2 + np.cos(phi) ** 2)

    return J


J = J(x, y)
print(J.shape)
plt.pcolormesh(x, y, np.absolute(J))
plt.colorbar()
plt.show()

plt.pcolormesh(x, y, np.absolute(np.fft.fft2(J)))
plt.colorbar()
plt.show()
