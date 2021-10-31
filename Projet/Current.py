import numpy as np
import matplotlib.pyplot as plt

# Equation parameters
I = 1
L = 0.1
lda = 3e8/10e9
k = 2*np.pi/lda
alpha = np.pi/4 # tilt angle of dipole

# meshgrid of XY plane
# reference point at the center of the square
x = np.linspace(0, 1, 100)
y = np.linspace(-0.5, 0.5, 100)
z = - 0.5  # use dipole as height reference
X, Y = np.meshgrid(x, y)

# cartesian to spherical coordinates
phi = np.arctan2(Y, X)
theta = np.arctan2(np.sqrt(X**2 + Y**2), z) + alpha  # dipole is tilted 45°
theta_prime = np.pi/2 - theta

# distance from dipole to plane
R = 0.5/np.cos(theta_prime + np.pi/4) * 1/np.cos(phi)

# induced current on X-Y plane surface
dir = np.sqrt((np.sqrt(2)/2 * np.sin(phi))**2 + np.cos(phi)**2)
J = 1j * I*L/lda * np.sin(theta) * np.sinc(k * L/2 * np.cos(theta)) * np.cos(theta)**2 * np.exp(-1j * k*R)/R * dir
J = np.absolute(J)


plt.pcolormesh(X, Y, J)
plt.colorbar()
plt.savefig("induced_current.svg")
plt.show()