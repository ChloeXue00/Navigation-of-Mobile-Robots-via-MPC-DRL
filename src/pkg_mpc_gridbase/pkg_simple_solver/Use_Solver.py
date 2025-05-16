import sys
import numpy as np
import matplotlib.pyplot as plt

# Use Direct Interface
# ------------------------------------
sys.path.insert(1, './trial_optimizer/navigation')
import navigation 

(nu, nx, N, L, ts) = (2, 3, 20, 0.5, 0.1)   
solver = navigation.solver()
result = solver.run(p=[-1.0, -2.0, 0.0],
                    initial_guess=[1] * (nu*N))
u_star = result.solution
print(u_star)


# Plot trajectory
# ------------------------------------
"""
x_init=[-10, -10, 0.0]
x_states = [0.0] * (nx*(N+2))
x_states[0:nx+1] = x_init
for t in range(0, N):
    u_t = u_star[t*nu:(t+1)*nu]

    x = x_states[t * nx]
    y = x_states[t * nx + 1]
    theta = x_states[t * nx + 2]

    theta_dot = (1/L) * (u_t[1] * np.cos(theta) - u_t[0] * np.sin(theta))

    x_states[(t+1)*nx] = x + ts * (u_t[0] + L*np.sin(theta)*theta_dot)
    x_states[(t+1)*nx+1] = y + ts * (u_t[1] - L*np.cos(theta)*theta_dot)
    x_states[(t+1)*nx+2] = theta + ts*theta_dot

xx = x_states[0:nx*N:nx]
xy = x_states[1:nx*N:nx]

plt.plot(xx, xy, '-o')
plt.show()"""


import numpy as np
import matplotlib.pyplot as plt

# Define circle parameters
radius = 1
theta = np.linspace(0, 2*np.pi, 100)
x_circle = radius * np.cos(theta)
y_circle = radius * np.sin(theta)

# Your existing code
x_init = [-1, -2, 0.0]
x_states = [0.0] * (nx*(N+2))
x_states[0:nx+1] = x_init
for t in range(0, N):
    u_t = u_star[t*nu:(t+1)*nu]

    x = x_states[t * nx]
    y = x_states[t * nx + 1]
    theta = x_states[t * nx + 2]

    theta_dot = (1/L) * (u_t[1] * np.cos(theta) - u_t[0] * np.sin(theta))

    x_states[(t+1)*nx] = x + ts * (u_t[0] + L*np.sin(theta)*theta_dot)
    x_states[(t+1)*nx+1] = y + ts * (u_t[1] - L*np.cos(theta)*theta_dot)
    x_states[(t+1)*nx+2] = theta + ts*theta_dot

xx = x_states[0:nx*N:nx]
xy = x_states[1:nx*N:nx]

# Plot the trajectory
plt.plot(xx, xy, '-o', label='Trajectory')

# Plot the circle
plt.plot(x_circle, y_circle, 'r--', label='Circle (r=3)')

# Additional plot settings
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory and Circle')
plt.axis('equal')  # Ensure aspect ratio is equal
plt.legend()  # Show legend
plt.grid(True)  # Show grid

# Show the plot
plt.show()


time = np.arange(0, ts*N, ts)
plt.plot(time, x_states[2:nx*N:nx])
plt.xlabel('Time')
plt.ylabel('Pose')
plt.show()







# Plot solution
# ------------------------------------
time = np.arange(0, ts*N, ts)
ux = u_star[0:nu*N:2]
uy = u_star[1:nu*N:2]

plt.subplot(211)
plt.plot(time, ux, '-o')
plt.ylabel('u_x')
plt.subplot(212)
plt.plot(time, uy, '-o')
plt.ylabel('u_y')
plt.xlabel('Time')
plt.show()

