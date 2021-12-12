import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.fft import fft, ifft, fft2, fftshift
from matplotlib.colors import Normalize

# param
lda = 3e8 / 10e9
I = 1
l = .1

# mesh_size
n = 100
# meshgrid
x = np.linspace(-.5, .5, n)  # définition du plan, centré en (0,0,0.5)
y = np.linspace(-.5, .5, n)


def H(x, y):
    p = np.array([np.sqrt(2) / 2, 0, np.sqrt(2) / 2])           # orientation du dipole
    q = np.array([np.sqrt(2) / 2, 0, -np.sqrt(2) / 2])          # perpendiculaire au dipole
    u = np.array([x + .5, y, -0.5])                             # vecteur U (la plaque se trouve en z = -0.5)

    R = np.linalg.norm(u)                                       # R = norme de U
    u_hat = u / np.linalg.norm(u)

    sintheta = np.linalg.norm(np.cross(u_hat, p))               # norme du cross product donne sin de l'angle
    costhetabis = np.dot(u_hat, q)                              # dot product donne le cos de l'angle
    k = 2 * np.pi / lda                                         # nombre d'onde

    E = 1j * l * I / lda * sintheta * np.exp(-1j * k * R) / R * costhetabis ** 8  # * np.sinc(k * l / 2 * costheta)
    E = E * (p - np.dot(p, u_hat) * u_hat)

    H = np.cross(u_hat, E)

    H = np.array([H[0], H[1], 0])

    return H


def AF(x, y, gt):
    u = np.array([x, y, np.sqrt(1 - x ** 2 - y ** 2)])  # u décrit une sphère np.sqrt(1-x**2 - y**2)
    u_hat = u / np.linalg.norm(u)
    F = gt - np.dot(gt, u_hat) * u_hat

    return F


def apply_shift(H, x, y):
    k = 2*np.pi/lda
    ux0 = np.sin(np.pi / 6)
    shiftx = np.exp(1j*k*x*ux0)
    return H*shiftx


Jv = np.zeros((n, n, 3), dtype=complex)
Jv_shift = np.zeros((n, n, 3), dtype=complex)

N = np.array([0, 0, 1])
for i in range(n):
    for j in range(n):
        Hv_shift = apply_shift(H(x[i], y[j]), x[i], y[j])
        Jv_shift[i][j] = 2 * np.cross(N, Hv_shift)
        Jv[i][j] = 2 * np.cross(N, H(x[i], y[j]))

a = 1 / n
ux = np.linspace(-lda / (2 * a), lda / (2 * a), n)
uy = np.linspace(-lda / (2 * a), lda / (2 * a), n)

g1 = fft2(np.transpose(Jv, (2, 0, 1))[0])
g1 = fftshift(g1)
g2 = fft2(np.transpose(Jv, (2, 0, 1))[1])
g2 = fftshift(g2)

F = np.zeros((n, n, 3), dtype=complex)
for i in range(n):
    for j in range(n):
        F[i][j] = AF(ux[i], uy[j], np.array([g1[i][j], g2[i][j], 0]))

# Courant
Jvx = np.transpose(Jv, (2, 0, 1))[0]
Jvy = np.transpose(Jv, (2, 0, 1))[1]

# Plot courant induit
Jv = np.absolute(Jv)
Jv = np.linalg.norm(Jv, axis=2)
X, Y = np.meshgrid(x, y, indexing='ij')
plt.pcolormesh(X, Y, Jv)
plt.colorbar()
plt.show()

Xtest, Ytest = np.meshgrid(x, y)

# Courant en vectoriel
plt.streamplot(Xtest, Ytest, np.absolute(Jvx), np.absolute(Jvy), density=1.4, linewidth=None, color='#A23BEC')
plt.show()

# FFT
UXTest, UYTest = np.meshgrid(ux, uy)
Fx = np.transpose(F, (2, 0, 1))[0]
Fy = np.transpose(F, (2, 0, 1))[1]
plt.streamplot(UXTest, UYTest, np.absolute(Fx), np.absolute(Fy), density=1.4, linewidth=None, color='#A23BEC')
plt.show()

F = np.absolute(F)
F = np.linalg.norm(F, axis=2)
UX, UY = np.meshgrid(ux, uy, indexing='ij')
plt.pcolormesh(UX, UY, F)
plt.colorbar()
plt.show()

# FFT shift
g1 = fft2(np.transpose(Jv_shift, (2, 0, 1))[0])
g1 = fftshift(g1)
g2 = fft2(np.transpose(Jv_shift, (2, 0, 1))[1])
g2 = fftshift(g2)

F = np.zeros((n, n, 3), dtype=complex)
for i in range(n):
    for j in range(n):
        F[i][j] = AF(ux[i], uy[j], np.array([g1[i][j], g2[i][j], 0]))

F = np.absolute(F)
F = np.linalg.norm(F, axis=2)
UX, UY = np.meshgrid(ux, uy, indexing='ij')
plt.pcolormesh(UX, UY, F)
plt.colorbar()
plt.show()