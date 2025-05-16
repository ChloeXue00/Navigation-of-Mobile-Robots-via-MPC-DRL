#THIS IS NOT HOW THE CURRENT SOLVER IS BUILT BUT THE PREVIOUS MEHTOD


from typing import Callable, Union

import casadi.casadi as cs # type: ignore
from opengen import opengen as og # type: ignore # or "import opengen as og"
import matplotlib.pyplot as plt
import numpy as np


from pkg_mpc_tracker.casadi_build import mpc_helper, mpc_cost
from configs import MpcConfiguration, CircularRobotSpecification

class PanocBuilder:
    """Build the MPC module via OPEN. Define states, inputs, cost, and constraints.

    Methods:
        build: Build the MPC problem and solver.
    """
    def __init__(self, mpc_config: MpcConfiguration, robot_config: CircularRobotSpecification):
        self.config = mpc_config
        self.robot_config = robot_config
        ### Frequently used
        self.ts = self.config.ts        # sampling time
        self.ns = self.config.ns        # number of states
        self.nu = self.config.nu        # number of inputs
        self.N_hor = self.config.N_hor  # control/pred horizon
        self.num_params_per_convex_hull = 12


    def build(self, motion_model:  Callable[[cs.SX, cs.SX, float], cs.SX]):
        """ Build the MPC problem and solver, including states, inputs, cost, and constraints.

        Args:
            motion_model: Callable function `s'=f(s,u,ts)` that generates next state given the current state and action.

        Notes:
            Inputs (u): speed, angular speed
            states (s): x, y, theta
            
        """

        u = cs.SX.sym('u', self.nu*self.N_hor)  # 0. Inputs from 0 to N_hor-1
        u_m1 = cs.SX.sym('u_m1', self.nu)       # 1. Input at kt=-1
        s_0 = cs.SX.sym('s_0', self.ns)         # 2. State at kt=0
        s_N = cs.SX.sym('s_N', self.ns)         # 3. State of goal at kt=N_hor
        q = cs.SX.sym('q', self.config.nq)      # 4. Penalty for terms related to states/inputs

        r_s = cs.SX.sym('r_s', self.ns*self.N_hor)  # 5. Reference states
        r_v = cs.SX.sym('r_v', self.N_hor)          # 6. Reference speed

        s_as = cs.SX.sym('s_as', 10* self.num_params_per_convex_hull)   #safe areas, needs to get fixed, 10 convex hulls              

        q_s_as = cs.SX.sym('q_s_as', self.N_hor)               # Convex hull weights


        z = cs.vertcat(u_m1, s_0, s_N, q, r_s, r_v, s_as, q_s_as) #parameters needed
        (x, y, theta) = (s_0[0], s_0[1], s_0[2])
        (x_goal, y_goal, theta_goal) = (s_N[0], s_N[1], s_N[2])
        (v_init, w_init) = (u_m1[0], u_m1[1])
        (qpos, qvel, qtheta, rv, rw)                    = (q[0], q[1], q[2], q[3], q[4])
        (qN, qthetaN, qrpd, acc_penalty, w_acc_penalty) = (q[5], q[6], q[7], q[8], q[9])
        
        ref_states = cs.reshape(r_s, (self.ns, self.N_hor)) # each column is a state
        ref_states = cs.horzcat(ref_states, ref_states[:,[-1]])[:2, :]

        cost = 0
        penalty_constraints = 1
        state = cs.vcat([x,y,theta])
        for kt in range(0, self.N_hor): # LOOP OVER PREDICTIVE HORIZON
            ### Run step with motion model
            u_t = u[kt*self.nu:(kt+1)*self.nu]
            state = motion_model(state, u_t, self.ts)

            ### Reference deviation costs
            cost += mpc_cost.cost_refpath_deviation(state, ref_states[:, kt:], weight=qrpd)   # [cost] reference path deviation
            cost += qvel * (u_t[0]-r_v[kt])**2                                                # [cost] refenrence velocity deviation
            cost += cs.sum1(cs.vertcat(rv, rw) * u_t**2)                                      # [cost] penalize control actions
            cost += qN*((state[0]-x_goal)**2 + (state[1]-y_goal)**2) + qthetaN*(state[2]-theta_goal)**2 # terminated cost



            # Cost outside valid convex hulls
            cost_convex = 1
            for i in range(10):
                eq_param = s_as[i*self.num_params_per_convex_hull : (i+1)*self.num_params_per_convex_hull]
                n_edges = int(self.num_params_per_convex_hull / 3) # 3 means b, a0, a1
                b, a0, a1 = eq_param[:n_edges], eq_param[n_edges:2*n_edges], eq_param[2*n_edges:]
                outside_convex = self.outside_cvx_polygon(state, b.T, a0.T, a1.T)
                penalty_constraints *= cs.fmin(0, cs.vertcat(outside_convex))   #         If prod(|min(0,all)|)>0, then the point is inside; Otherwise not.
                cost_convex *= self.cost_outside_cvx_polygon(state, b.T, a0.T, a1.T, weight=q_s_as[kt])         #If prod(|max(0,all)|)>0, then the point is outside all valid convex hulls; Otherwise it is inside one of them .

            cost +=cost_convex

                        ### Max speed bound
            umin = [self.robot_config.lin_vel_min, -self.robot_config.ang_vel_max] * self.N_hor
            umax = [self.robot_config.lin_vel_max,  self.robot_config.ang_vel_max] * self.N_hor
            bounds = og.constraints.Rectangle(umin, umax)


                        ### Acceleration bounds and cost
            v = u[0::2] # velocity
            w = u[1::2] # angular velocity
            acc   = (v-cs.vertcat(v_init, v[0:-1]))/self.ts
            w_acc = (w-cs.vertcat(w_init, w[0:-1]))/self.ts
            acc_constraints = cs.vertcat(acc, w_acc)


            acc_min   = [ self.robot_config.lin_acc_min] * self.N_hor 
            w_acc_min = [-self.robot_config.ang_acc_max] * self.N_hor
            acc_max   = [ self.robot_config.lin_acc_max] * self.N_hor
            w_acc_max = [ self.robot_config.ang_acc_max] * self.N_hor
            acc_bounds = og.constraints.Rectangle(acc_min + w_acc_min, acc_max + w_acc_max)

                        # Accelerations cost
            cost += cs.mtimes(acc.T, acc)*acc_penalty
            cost += cs.mtimes(w_acc.T, w_acc)*w_acc_penalty


            problem = og.builder.Problem(u, z, cost) \
                #.with_constraints(bounds) \
                #.with_aug_lagrangian_constraints(acc_constraints, acc_bounds)
            problem.with_penalty_constraints(penalty_constraints)


            build_config = og.config.BuildConfiguration()\
                .with_build_directory(self.config.build_directory)\
                .with_build_mode("debug")
            build_config.with_build_python_bindings()

            meta = og.config.OptimizerMeta()\
                .with_optimizer_name(self.config.optimizer_name)
            solver_config = og.config.SolverConfiguration()\
                            .with_initial_penalty(10) \
                            .with_max_duration_micros(500_000) # micros (0.5 sec)
            
            builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_config, solver_config) \
                .with_verbosity_level(1)
            
            builder.build()



        """
                # Consider the inequality: 0 ≤ b - (Ax + By)

        # If the point (x, y) is inside the convex hull:
        #   - Ax + By will be less than b
        #   - So, b - (Ax + By) will be positive
        #   - Thus, the inequality 0 ≤ b - (Ax + By) will hold

        # If the point (x, y) is outside the convex hull:
        #   - Ax + By will be greater than b
        #   - So, b - (Ax + By) will be negative
        #   - Thus, the inequality 0 ≤ b - (Ax + By) will not hold

        """

    def cost_outside_cvx_polygon(self, state: cs.SX, b: cs.SX, a0: cs.SX, a1: cs.SX, weight:Union[cs.SX, float]=1.0):
        """Cost (weighted squared) for being inside a convex polygon defined by `b - [a0,a1]*[x,y]' > 0`.
            
        Args:
            state: The (n*1)-dim target point.
            b:  Shape (1*m) with m half-space offsets.
            a0: Shape (1*m) with m half-space weight vectors.
            a1: Shape (1*m) with m half-space weight vectors.
            
        Returns:
            cost: The (1*1)-dim weighted square cost. If inside, return positive value, else return 0.

        Notes:
            Each half-space if defined as `b - [a0,a1]*[x,y]' > 0`.
            If prod(|max(0,all)|)>0, then the point is inside; Otherwise not.
        """
        indicator = self.outside_cvx_polygon(state, b, a0, a1)
        cost = weight * indicator**2
        return cost



    def outside_cvx_polygon(self, point: cs.SX, b: cs.SX, a0: cs.SX, a1: cs.SX) -> cs.SX:
        """Check if a point is outside a convex polygon defined by half-spaces.

        Args:
            point: The (n*1)-dim target point.
            b: Shape  (1*m) with m half-space offsets.
            a0: Shape (1*m) with m half-space weight vectors.
            a1: Shape (1*m) with m half-space weight vectors.

        Returns:
            is_outside: The (1*1)-dim indicator. If outside, return positive value, else return 0.

        Notes:
            Each half-space if defined as `b - [a0,a1]*[x,y]' > 0`.
            If sum(|min(0,all)|)>0, then the point is outside; Otherwise not.
        """
        eq_mtx: cs.SX = cs.vertcat(b, -a0, -a1)
        result: cs.SX = cs.mtimes(eq_mtx.T, cs.vertcat(1, point[0], point[1]))
        is_outside = 0
        for i in range(result.shape[0]):
            is_outside += cs.fmin(0, result[i]) ** 2
        return is_outside

