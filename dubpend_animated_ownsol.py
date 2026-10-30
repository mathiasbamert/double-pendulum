import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# start parameters 
g = 9.81
t_max = 10
h = 0.01
l1 = 1
l2 = 1
m1 = 1
m2 = 1
theta1_0 = np.pi / 3
theta2_0 = np.pi * 3/4
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
  k2 = derivs(l1, l2, m1, m2, y + h/2 * k1) 
  k3 = derivs(l1, l2, m1, m2, y + h/2 * k2)
  k4 = derivs(l1, l2, m1, m2, y + h * k3)
  y = y + h * (k1 / 6 + k2 / 3 + k3 / 3  + k4 / 6)
  return y

### Set coordinate system
fig, ax = plt.subplots()

ax.set_xlim(-1.2 * (l1 + l2), 1.2 * (l1 + l2))
ax.set_ylim(-1.2 * (l1 + l2), 0.2 * (l1 + l2))

ax.set_aspect("equal")
ax.grid()

#pendulum rod(s)
line1, = ax.plot([],[], "o-", lw= 2, markersize = 10)
line2, = ax.plot([], [], "o-", lw = 2, markersize = 10)
xtraj = []
ytraj = []
traj, = ax.plot([],[], 'o', markersize = 1)

### initialization function
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2,

y = np.array([theta1_0, omega1_0, theta2_0, omega2_0])
results = []
times = np.arange(0, t_max, h)
for _ in times: 
  results.append(y.copy())
  y = RK4_step(l1, l2, m1, m2, y, h)
results = np.array(results)

def update(frame): 
  theta1, theta2 = results[frame, 0::2]

  #transform from polar to cartesian coordinates
  x1 = l1 * np.sin(theta1)
  y1 = -l1 * np.cos(theta1)

  x2 = l1 * np.sin(theta1) + l2 * np.sin(theta2)
  y2 = -(l1 * np.cos(theta1) + l2 * np.cos(theta2))

  line1.set_data([0, x1], [0, y1])

  line2.set_data([x1, x2], [y1, y2])

  xtraj.append(x2)
  ytraj.append(y2)
  traj.set_data(xtraj, ytraj)

  return line1, line2, traj


# start animation 
ani = FuncAnimation(
    fig,
    update,
    frames=len(results),
    init_func=init,
    interval = 10,  # milliseconds
    blit=True
)

plt.show()


#Energy Conservation 
theta1, omega1, theta2, omega2 = results.T

v1 = l1 * omega1
v2 = np.sqrt(l1**2 * omega1**2 + l2**2 * omega2**2 + 2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2))
T = 1/2 * m1 * v1**2 + 1/2 * m2 * v2**2

y1 = -l1 * np.cos(theta1) 
y2 = -l1 * np.cos(theta1) - l2 * np.cos(theta2)

V = g * (m1 * y1 + m2 * y2)

E = T + V 

E0 = E[0]
abs_error = E - E0 
rel_error = abs_error / abs(E0)

fig2 = plt.figure()

ax2 = fig2.add_subplot()

ax2.plot(times, rel_error)

plt.show()

print("Initial energy:", E0)
print("Maximum absolute error:", np.max(np.abs(abs_error)))
print("Maximum relative error:", np.max(np.abs(rel_error)))

