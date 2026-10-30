import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
 
# PHYSICS / SOLVER
 
g = 9.81
t_max = 20
h = 0.002
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
 
 
# VISUALIZATION

 
class DoublePendulumAnimator:

    """Owns the pendulum animation figure: rods, trailing path, time readout."""

    def __init__(self, times, results, l1, l2, ax, color="tab:red", label="", trail_length=400):
        self.times = times     
        self.results = results
        self.l1 = l1
        self.l2 = l2
        self.ax = ax                       # <-- passed in, not created here
        self.trail = deque(maxlen=trail_length)

        self.rod1, = self.ax.plot([], [], "o-", lw=2, markersize=10,
                                   color="tab:blue", alpha=0.6)
        self.rod2, = self.ax.plot([], [], "o-", lw=2, markersize=10,
                                   color=color, label=label)
        self.trail_line, = self.ax.plot([], [], lw=1, color=color, alpha=0.4)

    def init(self):
        self.rod1.set_data([], [])
        self.rod2.set_data([], [])
        self.trail_line.set_data([], [])
        return self.rod1, self.rod2, self.trail_line

    def update(self, frame):
        theta1, theta2 = self.results[frame, 0], self.results[frame, 2]
        x1 = self.l1 * np.sin(theta1)
        y1 = -self.l1 * np.cos(theta1)
        x2 = x1 + self.l2 * np.sin(theta2)
        y2 = y1 - self.l2 * np.cos(theta2)

        self.rod1.set_data([0, x1], [0, y1])
        self.rod2.set_data([x1, x2], [y1, y2])
        self.trail.append((x2, y2))
        tx, ty = zip(*self.trail)
        self.trail_line.set_data(tx, ty)
        return self.rod1, self.rod2, self.trail_line
 
 
def plot_energy_diagnostics(times, results, l1, l2, m1, m2, g, h):
    """Two labeled panels: energy components, and relative conservation error."""
    theta1, omega1, theta2, omega2 = results.T
 
    v1 = l1 * omega1
    v2 = np.sqrt(l1**2 * omega1**2 + l2**2 * omega2**2
                 + 2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2))
    T = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
 
    y1 = -l1 * np.cos(theta1)
    y2 = -l1 * np.cos(theta1) - l2 * np.cos(theta2)
    V = g * (m1 * y1 + m2 * y2)
 
    E = T + V
    E0 = E[0]
    abs_error = E - E0
    rel_error = abs_error / abs(E0)
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
 
    ax1.plot(times, T, label="Kinetic", color="tab:orange")
    ax1.plot(times, V, label="Potential", color="tab:green")
    ax1.plot(times, E, label="Total", color="k", lw=2)
    ax1.set_ylabel("Energy (J)")
    ax1.set_title("Energy components")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
 
    ax2.plot(times, np.abs(rel_error), color="tab:red")
    ax2.set_yscale("log")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("|Relative error|")
    ax2.set_title(f"Energy conservation error (RK4, h={h})")
    ax2.grid(alpha=0.3, which="both")
 
    fig.tight_layout()
 
    print("Initial energy:", E0)
    print("Maximum absolute error:", np.max(np.abs(abs_error)))
    print("Maximum relative error:", np.max(np.abs(rel_error)))
 
    return fig, (ax1, ax2)
 
 
# MAIN
 
if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(6, 6))

    reach = 1.2 * (l1 + l2)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, 0.2 * reach)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Two double pendulums, nearby initial conditions")

    y0 = np.array([theta1_0, omega1_0, theta2_0, omega2_0])
    y0_2 = np.array([theta1_0 + 0.001, omega1_0, theta2_0 + 0.001, omega2_0])
    times1, results1 = run_simulation(l1, l2, m1, m2, y0, t_max, h)
    times2, results2 = run_simulation(l1,l2,m1,m2, y0_2, t_max, h)

    p1 = DoublePendulumAnimator(times1, results1, l1, l2, ax, color="tab:red",  label="θ₁₀ = 60°")
    p2 = DoublePendulumAnimator(times2, results2, l1, l2, ax, color="tab:green", label="θ₁₀ = 60.001°")
    ax.legend(loc="upper right")

    def init():
        return (*p1.init(), *p2.init())

    def update(frame):
        return (*p1.update(frame), *p2.update(frame))

    interval = 20
    frame_step = 10
    frames = range(0, len(results1), frame_step)

    ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                        interval=interval, blit=True)

    plt.show()
    plot_energy_diagnostics(times1, results1, l1, l2, m1, m2, g, h)
    plot_energy_diagnostics(times2, results2, l1, l2, m1, m2, g, h)
    plt.show()
    