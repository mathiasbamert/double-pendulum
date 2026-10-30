"""Double pendulum simulation with a custom RK4 integrator.

Two pendulums with nearly identical initial conditions are integrated and
animated side by side to show sensitive dependence on initial conditions.

Equations of motion
-------------------
Angles theta1, theta2 are measured from the downward vertical.
With delta = theta1 - theta2, k = m2 / (m1 + m2) and c = l2 / l1:

    theta1'' + c*k*cos(delta) * theta2'' = -(g/l1) sin(theta1) - k*c*sin(delta) * omega2^2
    (l1/l2)*cos(delta) * theta1'' + theta2'' = (l1/l2) sin(delta) * omega1^2 - (g/l2) sin(theta2)

This 2x2 linear system is solved for (theta1'', theta2'') at every step and the
resulting first-order system is advanced with classical 4th-order Runge-Kutta.

Usage
-----
    python double_pendulum.py                    # interactive animation
    python double_pendulum.py --save docs        # write docs/chaos.gif + docs/energy.png
    python double_pendulum.py --help             # all options
"""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

G = 9.81


# ----------------------------------------------------------------------------
# Physics / solver
# ----------------------------------------------------------------------------

def derivs(l1, l2, m1, m2, g, y):
    """Time derivative of the state y = [theta1, omega1, theta2, omega2]."""
    theta1, omega1, theta2, omega2 = y

    k = m2 / (m1 + m2)
    c = l2 / l1
    delta = theta1 - theta2

    a1 = c * k * np.cos(delta)
    a2 = l1 / l2 * np.cos(delta)
    det_a = 1 - a1 * a2

    f1 = -(g / l1 * np.sin(theta1) + k * c * np.sin(delta) * omega2**2)
    f2 = 1 / c * np.sin(delta) * omega1**2 - g / l2 * np.sin(theta2)

    alpha1 = (f1 - a1 * f2) / det_a
    alpha2 = (f2 - a2 * f1) / det_a

    return np.array([omega1, alpha1, omega2, alpha2])


def rk4_step(l1, l2, m1, m2, g, y, h):
    """Advance the state y by one RK4 step of size h."""
    k1 = derivs(l1, l2, m1, m2, g, y)
    k2 = derivs(l1, l2, m1, m2, g, y + h / 2 * k1)
    k3 = derivs(l1, l2, m1, m2, g, y + h / 2 * k2)
    k4 = derivs(l1, l2, m1, m2, g, y + h * k3)
    return y + h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)


def run_simulation(l1, l2, m1, m2, g, y0, t_max, h):
    """Integrate from y0 over [0, t_max). Returns (times, results)."""
    times = np.arange(0, t_max, h)
    y = np.array(y0, dtype=float)
    results = np.empty((len(times), 4))
    for i in range(len(times)):
        results[i] = y
        y = rk4_step(l1, l2, m1, m2, g, y, h)
    return times, results


def total_energy(results, l1, l2, m1, m2, g):
    """Return (T, V, E) arrays for a trajectory."""
    theta1, omega1, theta2, omega2 = results.T

    v1_sq = (l1 * omega1) ** 2
    v2_sq = (l1**2 * omega1**2 + l2**2 * omega2**2
             + 2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2))
    T = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq

    y1 = -l1 * np.cos(theta1)
    y2 = y1 - l2 * np.cos(theta2)
    V = g * (m1 * y1 + m2 * y2)
    return T, V, T + V


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------

class DoublePendulumAnimator:
    """Draws one pendulum (two rods plus a trailing path) on a given axes."""

    def __init__(self, results, l1, l2, ax, color="tab:red", label="", trail_length=400):
        self.results = results
        self.l1 = l1
        self.l2 = l2
        self.trail = deque(maxlen=trail_length)

        self.rod1, = ax.plot([], [], "o-", lw=2, markersize=8, color=color, alpha=0.6)
        self.rod2, = ax.plot([], [], "o-", lw=2, markersize=8, color=color, label=label)
        self.trail_line, = ax.plot([], [], lw=1, color=color, alpha=0.4)

    def init(self):
        self.trail.clear()
        for artist in (self.rod1, self.rod2, self.trail_line):
            artist.set_data([], [])
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
    """Two panels: energy components, and relative energy conservation error."""
    T, V, E = total_energy(results, l1, l2, m1, m2, g)
    E0 = E[0]
    rel_error = (E - E0) / abs(E0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(times, T, label="Kinetic", color="tab:orange")
    ax1.plot(times, V, label="Potential", color="tab:green")
    ax1.plot(times, E, label="Total", color="k", lw=2)
    ax1.set_ylabel("Energy (J)")
    ax1.set_title("Energy components")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    ax2.plot(times, np.abs(rel_error) + 1e-18, color="tab:red")  # avoid log(0)
    ax2.set_yscale("log")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("|Relative error|")
    ax2.set_title(f"Energy conservation error (RK4, h={h})")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()

    print(f"Initial energy:          {E0:.6f} J")
    print(f"Maximum relative error:  {np.max(np.abs(rel_error)):.3e}")
    return fig


def plot_divergence(times, results1, results2):
    """Angle difference of the second pendulum arm between the two runs."""
    diff = np.abs(results1[:, 2] - results2[:, 2])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.semilogy(times, diff + 1e-18, color="tab:purple")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$|\theta_2^{(1)} - \theta_2^{(2)}|$ (rad)")
    ax.set_title("Divergence of the two trajectories")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Double pendulum chaos demo (RK4).")
    p.add_argument("--t-max", type=float, default=20.0, help="simulated seconds")
    p.add_argument("--h", type=float, default=0.002, help="RK4 time step (s)")
    p.add_argument("--l1", type=float, default=1.0, help="length of rod 1 (m)")
    p.add_argument("--l2", type=float, default=1.0, help="length of rod 2 (m)")
    p.add_argument("--m1", type=float, default=1.0, help="mass 1 (kg)")
    p.add_argument("--m2", type=float, default=1.0, help="mass 2 (kg)")
    p.add_argument("--theta1", type=float, default=60.0, help="initial theta1 (deg)")
    p.add_argument("--theta2", type=float, default=135.0, help="initial theta2 (deg)")
    p.add_argument("--omega1", type=float, default=1.0, help="initial omega1 (rad/s)")
    p.add_argument("--omega2", type=float, default=2.0, help="initial omega2 (rad/s)")
    p.add_argument("--delta", type=float, default=0.001,
                   help="perturbation added to theta1 of the second pendulum (deg)")
    p.add_argument("--fps", type=int, default=30, help="animation frame rate")
    p.add_argument("--save", metavar="DIR", default=None,
                   help="save chaos.gif, energy.png and divergence.png to DIR")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    return p.parse_args()


def main():
    args = parse_args()
    l1, l2, m1, m2, h = args.l1, args.l2, args.m1, args.m2, args.h

    y0_a = np.array([np.radians(args.theta1), args.omega1,
                     np.radians(args.theta2), args.omega2])
    y0_b = y0_a.copy()
    y0_b[0] += np.radians(args.delta)

    times, res_a = run_simulation(l1, l2, m1, m2, G, y0_a, args.t_max, h)
    _, res_b = run_simulation(l1, l2, m1, m2, G, y0_b, args.t_max, h)

    fig, ax = plt.subplots(figsize=(6, 6))
    reach = 1.2 * (l1 + l2)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, 0.2 * reach)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Two double pendulums, nearby initial conditions")

    p1 = DoublePendulumAnimator(res_a, l1, l2, ax, "tab:red",
                                f"θ₁ = {args.theta1:g}°")
    p2 = DoublePendulumAnimator(res_b, l1, l2, ax, "tab:green",
                                f"θ₁ = {args.theta1 + args.delta:g}°")
    ax.legend(loc="upper right")

    def init():
        return (*p1.init(), *p2.init())

    def update(frame):
        return (*p1.update(frame), *p2.update(frame))

    # Pick a frame step so the animation plays in real time at the chosen fps.
    frame_step = max(1, round(1 / (args.fps * h)))
    frames = range(0, len(res_a), frame_step)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                        interval=1000 / args.fps, blit=True)

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        print(f"Saving {out / 'chaos.gif'} ...")
        ani.save(out / "chaos.gif", writer=PillowWriter(fps=args.fps), dpi=80)
        plot_energy_diagnostics(times, res_a, l1, l2, m1, m2, G, h).savefig(
            out / "energy.png", dpi=150)
        plot_divergence(times, res_a, res_b).savefig(out / "divergence.png", dpi=150)
    else:
        plot_energy_diagnostics(times, res_a, l1, l2, m1, m2, G, h)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()