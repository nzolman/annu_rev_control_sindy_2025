# most models are just directly taken from pysindy.odes, a local copy is provided for convenience.

import numpy as np

# Lorenz model
def lorenz(t, x, sigma=10, beta=2.66667, rho=28):
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]
    
# # Lotka model
# def lotka(t, x, p=[1., 4.]):
#     return [p[0] * x[0] - p[1] * x[0] * x[1], 
#             p[1] * x[0] * x[1] - 2 * p[0] * x[1]]
# Lotka model
def lotka(t, x, p=[2./3]):
    return [p[0] * x[0] - 2.0 *p[0]* x[0] * x[1], 
            1.0 * x[0] * x[1] - 1.0 * x[1]]


# Rossler model
def rossler(t, x, p=[0.2, 0.2, 5.7]):
    return [-x[1] - x[2], 
            x[0] + p[0] * x[1], 
            p[1] + (x[0] - p[2]) * x[2]]
    
# Van der Pol ODE
def van_der_pol(t, x, p=[0.5]):
    return [x[1], p[0] * (1 - x[0] ** 2) * x[1] - x[0]]

# Duffing ODE
def duffing(t, x, p=[0.2, 0.05, 1]):
    return [x[1], -p[0] * x[1] - p[1] * x[0] - p[2] * x[0] ** 3]

# Hopf bifurcation model
# def hopf(t, x, mu=-0.05, omega=1, A=1):
def hopf(t, x, mu=0.25, omega=1, A=0.25):
    return [
        mu * x[0] - omega * x[1] - A * x[0] * (x[0] ** 2 + x[1] ** 2),
        omega * x[0] + mu * x[1] - A * x[1] * (x[0] ** 2 + x[1] ** 2),
    ]
    
# Cart on a pendulum
def pendulum_on_cart(t, x, m=1, M=1, L=1, F=0, g=9.81):
    return [
        x[2],
        x[3],
        (
            (M + m) * g * np.sin(x[0])
            - F * np.cos(x[0])
            - m * L * np.sin(x[0]) * np.cos(x[0]) * x[2] ** 2
        )
        / (L * (M + m * np.sin(x[0]) ** 2)),
        (m * L * np.sin(x[0]) * x[2] ** 2 + F - m * g * np.sin(x[0]) * np.cos(x[0]))
        / (M + m * np.sin(x[0]) ** 2),
    ]
    
# Infamous double pendulum problem (frictionless if k1=k2=0)
def double_pendulum(
    t,
    x,
    m1=0.2704,
    m2=0.2056,
    a1=0.191,
    a2=0.1621,
    L1=0.2667,
    L2=0.2667,
    I1=0.003,
    I2=0.0011,
    g=9.81,
    k1=0,
    k2=0,
):
    return [
        x[2],
        x[3],
        (
            L1 * a2**2 * g * m2**2 * np.sin(x[0])
            - 2 * L1 * a2**3 * x[3] ** 2 * m2**2 * np.sin(x[0] - x[1])
            + 2 * I2 * L1 * g * m2 * np.sin(x[0])
            + L1 * a2**2 * g * m2**2 * np.sin(x[0] - 2 * x[1])
            + 2 * I2 * a1 * g * m1 * np.sin(x[0])
            - (L1 * a2 * x[2] * m2) ** 2 * np.sin(2 * (x[0] - x[1]))
            - 2 * I2 * L1 * a2 * x[3] ** 2 * m2 * np.sin(x[0] - x[1])
            + 2 * a1 * a2**2 * g * m1 * m2 * np.sin(x[0])
        )
        / (
            2 * I1 * I2
            + (L1 * a2 * m2) ** 2
            + 2 * I2 * L1**2 * m2
            + 2 * I2 * a1**2 * m1
            + 2 * I1 * a2**2 * m2
            - (L1 * a2 * m2) ** 2 * np.cos(2 * (x[0] - x[1]))
            + 2 * (a1 * a2) ** 2 * m1 * m2
        ),
        (
            a2
            * m2
            * (
                2 * I1 * g * np.sin(x[1])
                + 2 * L1**3 * x[2] ** 2 * m2 * np.sin(x[0] - x[1])
                + 2 * L1**2 * g * m2 * np.sin(x[1])
                + 2 * I1 * L1 * x[2] ** 2 * np.sin(x[0] - x[1])
                + 2 * a1**2 * g * m1 * np.sin(x[1])
                + L1**2 * a2 * x[3] ** 2 * m2 * np.sin(2 * (x[0] - x[1]))
                + 2 * L1 * a1**2 * x[2] ** 2 * m1 * np.sin(x[0] - x[1])
                - 2 * L1**2 * g * m2 * np.cos(x[0] - x[1]) * np.sin(x[0])
                - 2 * L1 * a1 * g * m1 * np.cos(x[0] - x[1]) * np.sin(x[0])
            )
        )
        / (
            2
            * (
                I1 * I2
                + (L1 * a2 * m2) ** 2
                + I2 * L1**2 * m2
                + I2 * a1**2 * m1
                + I1 * a2**2 * m2
                - (L1 * a2 * m2) ** 2 * np.cos(x[0] - x[1]) ** 2
                + a1**2 * a2**2 * m1 * m2
            )
        ),
    ]