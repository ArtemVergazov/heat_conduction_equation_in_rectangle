"""
solution of thermal conductivity equation:
du/dt = delta(u), 0 < x < 1, 0 < y < 2, 0 < t <= T

with Dirichlet boundary conditions:
u(0, y, t) = 0
u(1, y, t) = 0
u(x, 0, t) = 0
u(x, 2, t) = 0

and initial condition:
u(x, y, 0) = sin(2 * pi * x) * sin(pi * y) = u0(x)
"""

import numpy as np
from numpy import pi, exp, sin, cos, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# set equation parameters
Lx = 1.
Ly = 2.
T = 0.1

N = 50  # number of X intervals
M = 50  # number of Y intervals
J = 50  # number of time intervals

hx = Lx / N  # grid step that gives O(hx ** 2) error for Neumann boundary condition
hy = Ly / M
tau = T / J

gamma_x = tau / hx ** 2
gamma_y = tau / hy ** 2

x = np.linspace(0, Lx, N + 1)
y = np.linspace(0, Ly, M + 1)
t = np.linspace(0, T, J + 1)

xmesh2d, ymesh2d = np.meshgrid(x, y, indexing='ij')
tmesh3d, xmesh3d, ymesh3d = np.meshgrid(t, x, y, indexing='ij')


def analytical_solution(x, y, t):
    return sin(2 * pi * x) * sin(pi * y) * exp(-5 * pi ** 2 * t)


def u0(x, y):
    # initial condition
    return sin(2 * pi * x) * sin(pi * y)


def tridiag_mat_solver(A, F):
    # solve AY = -F, where A, Y, F - np.arrays, A - tridiagonal matrix

    n = len(F) - 1

    alpha = np.zeros(n + 1)
    beta = np.zeros(n + 1)

    a = np.zeros(n + 1)
    a[0] = 0
    for i in range(1, n + 1):
        a[i] = A[i, i - 1]
    b = np.zeros(n + 1)
    b[n] = 0
    for i in range(n):
        b[i] = A[i, i + 1]
    c = np.zeros(n + 1)
    for i in range(n + 1):
        c[i] = -A[i, i]

    alpha[0] = b[0] / c[0]
    beta[0] = F[0] / c[0]
    for i in range(1, n):
        alpha[i] = b[i] / (c[i] - a[i] * alpha[i - 1])
        beta[i] = (a[i] * beta[i - 1] + F[i]) / (c[i] - a[i] * alpha[i - 1])

    y = np.zeros(n + 1)
    y[n] = (F[n] + a[n] * beta[n - 1]) / (c[n] - a[n] * alpha[n - 1])
    for i in range(n - 1, -1, -1):
        y[i] = alpha[i] * y[i + 1] + beta[i]
    return y


def solve():
    w = np.zeros((2 * J + 1, N + 1, M + 1))

    # initial condition
    w[0, :, :] = u0(xmesh2d, ymesh2d)

    for j in range(0, 2 * J, 2):

        # from j to j + 1 / 2 -> solve system of N + 1 equations
        for m in range(1, M):
            A = np.zeros((N + 1, N + 1))
            F = np.zeros(N + 1)
            for n in range(1, N):
                A[n, n] = -(gamma_x + 1)
                A[n, n - 1] = 0.5 * gamma_x
                A[n, n + 1] = A[n, n - 1]
                F[n] = w[j, n, m] + 0.5 * gamma_y * (w[j, n, m + 1] - 2 * w[j, n, m] + w[j, n, m - 1])
            A[0, 0] = -1
            F[0] = 0  # boundary conditions
            A[-1, -1] = -1
            F[-1] = 0  # boundary conditions

            w[j + 1, :, m] = tridiag_mat_solver(A, F)

        # from j + 1 / 2 to j + 1 -> solve system of M + 1 equations
        for n in range(1, N):
            A = np.zeros((M + 1, M + 1))
            F = np.zeros(M + 1)
            for m in range(1, M):
                A[m, m] = -(gamma_y + 1)
                A[m, m - 1] = 0.5 * gamma_y
                A[m, m + 1] = A[m, m - 1]
                F[m] = w[j+1, n, m] + 0.5 * gamma_x * (w[j+1, n+1, m] - 2 * w[j+1, n, m] + w[j+1, n-1, m])
            A[0, 0] = -1
            F[0] = 0  # boundary conditions
            A[-1, -1] = -1
            F[-1] = 0  # boundary conditions

            w[j + 2, n, :] = tridiag_mat_solver(A, F)

        w[j + 2, 0, :] = 0
        w[j + 2, -1, :] = 0

    return w


def surf(ax, xmesh, ymesh, z, zlabel='', title=''):
    ax.plot_surface(xmesh, ymesh, z, cmap='winter', edgecolor='None')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()


w = solve()[::2, :, :]

u = analytical_solution(xmesh3d, ymesh3d, tmesh3d)
err = u - w

for j in range(0, J + 1, int(J / 4)):
    ax = plt.figure().add_subplot(111, projection='3d')
    surf(ax, xmesh2d, ymesh2d, u[j, :, :], 'u', 'analytical solution, t = ' + str(t[j]))
    ax = plt.figure().add_subplot(111, projection='3d')
    surf(ax, xmesh2d, ymesh2d, w[j, :, :], 'w', 'numerical solution, t = ' + str(t[j]))
    ax = plt.figure().add_subplot(111, projection='3d')
    surf(ax, xmesh2d, ymesh2d, err[j, :, :], 'err', 'error, t = ' + str(t[j]))
