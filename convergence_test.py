import numpy as np
import matplotlib.pyplot as plt
from double_pendulum import (
    derivs, RK4_step, run_simulation,
    l1, l2, m1, m2, g,
    theta1_0, theta2_0, omega1_0, omega2_0,
)
def energy(results, l1, l2, m1, m2, g): 
    theta1, omega1, theta2, omega2 = results.T
    v1 = l1 * omega1
    v2 = np.sqrt(l1**2 * omega1**2 + l2**2 * omega2**2
                  + 2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2))
    T = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
  
    y1 = -l1 * np.cos(theta1)
    y2 = -l1 * np.cos(theta1) - l2 * np.cos(theta2)
    V = g * (m1 * y1 + m2 * y2)
  
    return T + V
    
   
def energy_convergence_test(l1, l2, m1, m2, y0, g, t_max, h_steps): 
    E = []
    max_rel_errors = []
    for i,h in enumerate(h_steps): 
      times, results = run_simulation(l1, l2, m1, m2, y0, t_max, h)
      E_h = energy(results, l1, l2, m1, m2, g)
      E.append(E_h)
      rel_error = np.max(np.abs((E_h - E_h[0]) / E_h[0]))
      max_rel_errors.append(rel_error)
    return E, max_rel_errors

## data visualization

if __name__ == "__main__": 
  t_max = 15
  h_values = np.array([ 0.2, 0.15, 0.1, 0.0225, 0.02, 0.0175, 0.015, 0.0125, 0.01, 0.005, 0.002, 0.001])
  y0 = np.array([theta1_0, omega1_0, theta2_0, omega2_0])
  E, energy_errors = energy_convergence_test(l1, l2, m1, m2, y0, g, t_max, h_values)

  fig = plt.figure()
  ax = fig.add_subplot()

  ax.loglog(h_values, energy_errors)
  ref_line = energy_errors[-1] * (h_values / h_values[-1]) ** 4
  ax.loglog(h_values, ref_line, "k--", label="slope 4 (theoretical RK4)")
  ax.legend()
  plt.show()
  print(energy_errors)
  
  



  