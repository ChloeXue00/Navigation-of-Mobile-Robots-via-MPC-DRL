import casadi.casadi as cs
import opengen as og
import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
# Finally, CasADi is not an “optimal control problem solver”, that allows the user to enter an OCP and then gives the solution back. 
#Instead, it tries to provide the user with a set of “building blocks” that can be used to implement general-purpose or specific-purpose OCP solvers efficiently with a modest programming effort.

#This is just a trial example to get to know how to used OpenGen and Casadi 

# Build parametric optimizer
# ------------------------------------
(nu, nx, N, L, ts) = (2, 3, 20, 0.5, 0.1)
(xref, yref, thetaref) = (2, 2, 0)
(q, qtheta, r, qN, qthetaN) = (22, 0.1, 1, 140, 2)

u = cs.SX.sym('u', nu*N)
z0 = cs.SX.sym('z0', nx)

(x, y, theta) = (z0[0], z0[1], z0[2]) #starting states and these are the inputs

cost = 0
"""
c=0
for t in range(0, nu*N, nu):
    cost += q*((x-xref)**2 + (y-yref)**2) + qtheta*(theta-thetaref)**2
    u_t = u[t:t+2]
    theta_dot = (1/L) * (u_t[1] * cs.cos(theta) - u_t[0] * cs.sin(theta))
    cost += r * cs.dot(u_t, u_t)
    x += ts * (u_t[0] + L * cs.sin(theta) * theta_dot)
    y += ts * (u_t[1] - L * cs.cos(theta) * theta_dot)
    theta += ts * theta_dot
    c += cs.fmax(0, 1 - x**2 - y**2)

"""
c = 0
for t in range(0, N):
    cost += q*((x-xref)**2 + (y-yref)**2) + qtheta*(theta-thetaref)**2
    u_t = u[t*nu:(t+1)*nu]
    theta_dot = (1/L) * (u_t[1] * cs.cos(theta) - u_t[0] * cs.sin(theta))
    cost += r * cs.dot(u_t, u_t)
    x += ts * (u_t[0] + L * cs.sin(theta) * theta_dot)
    y += ts * (u_t[1] - L * cs.cos(theta) * theta_dot)
    theta += ts * theta_dot
    # ADD THIS LINE:
    c += cs.fmax(0, 1- (x**2 - y**2))


cost += qN*((x-xref)**2 + (y-yref)**2) + qthetaN*(theta-thetaref)**2

umin = [-3.0] * (nu*N)
umax = [3.0] * (nu*N)
bounds = og.constraints.Rectangle(umin, umax)

problem = og.builder.Problem(u, z0, cost).with_constraints(bounds).with_penalty_constraints(c)        

build_config = og.config.BuildConfiguration()\
    .with_build_directory("trial_optimizer")\
    .with_build_mode("debug")\
    .with_build_python_bindings()

meta = og.config.OptimizerMeta()\
    .with_optimizer_name("navigation")

solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-3)\
    .with_initial_tolerance(1e-3)\
    .with_max_outer_iterations(5)\
    .with_delta_tolerance(1e-2)\
    .with_penalty_weight_update_factor(50)\
    .with_initial_penalty(20.0)

builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config)
builder.build()

#python3 src/pkg_mpc_gridbase/pkg_simpleExample/Simple_Ex.py