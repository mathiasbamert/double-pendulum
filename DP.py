import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
 
# PHYSICS / SOLVER
 
g = 9.81
t_max = 20
h = 0.2
l1 = 1
l2 = 1
m1 = 1
m2 = 1
theta1_0 = np.pi / 3
theta2_0 = np.pi * 3 / 4
omega1_0 = 1
omega2_0 = 2
 
 
def derivs(l1, l2, m1, m2, y):
    theta1, omega1, theta2, omega2 = y
 
    k = m2 / (m1 + m2)
    c = l2 / l1
    delta = theta1 - theta2
 
    a1 = c * k * np.cos(delta)
    a2 = l1 / l2 * np.cos(delta)
    detA = (1 - a1 * a2)
 
    f1 = -(g / l1 * np.sin(theta1) + k * c * np.sin(delta) * omega2**2)
    f2 = 1 / c * np.sin(delta) * omega1**2 - g / l2 * np.sin(theta2)
 
    g1 = 1 / detA * (f1 - a1 * f2)
    g2 = 1 / detA * (f2 - a2 * f1)
 
    dtheta1_dt = omega1
    domega1_dt = g1
    dtheta2_dt = omega2
    domega2_dt = g2
 
    return np.array([dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt])
 
 
def RK4_step(l1, l2, m1, m2, y, h):
    k1 = derivs(l1, l2, m1, m2, y)
    k2 = derivs(l1, l2, m1, m2, y + h / 2 * k1)
    k3 = derivs(l1, l2, m1, m2, y + h / 2 * k2)
    k4 = derivs(l1, l2, m1, m2, y + h * k3)
    y = y + h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
    return y
 
 
def run_simulation(l1, l2, m1, m2, y0, t_max, h):
    """Runs the RK4 integration and returns (times, results)."""
    times = np.arange(0, t_max, h)
    y = y0.copy()
    results = np.empty((len(times), 4))
    for i in range(len(times)):
        results[i] = y
        y = RK4_step(l1, l2, m1, m2, y, h)
    return times, results

y0 = np.array([theta1_0, omega1_0, theta2_0, omega2_0])

h_values = np.arange(0.1, 2.2628, 0.01)

max = []

for i, h_ in enumerate(h_values): 
    _, results = run_simulation(l1, l2, m1, m2, y0, t_max, h_)
    max_val = np.max(np.abs(results), axis = 0)
    max.append(max_val[3])

print(max)


