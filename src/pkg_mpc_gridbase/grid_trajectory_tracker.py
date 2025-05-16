# System import
import os
import sys
import math
from typing import Callable, List, Optional

# External import
import numpy as np
from scipy.spatial import ConvexHull # type: ignore
import casadi.casadi as cs # type: ignore

# Custom import 
from configs import MpcConfiguration, CircularRobotSpecification

PathNode = tuple[float, float]


class TrajectoryTracker:
    """Generate a smooth trajectory tracking based on the reference path and obstacle information.

    Raises:
        TypeError: All states should be numpy.ndarry.
        ModuleNotFoundError: If the work mode is not supported.
        RuntimeError: If the solver cannot be run.
        ModuleNotFoundError: If the solver type is not supported.

    Attributes:
        config: MPC configuration.
        robot_spec: Robot specification.

    Methods:
        load_motion_model: Load the motion model (dynamics) of the robot.
        load_init_states: Load the initial state and goal state.
        set_work_mode: Set the basic work mode (base speed and weight parameters) of the MPC solver.
        set_current_state: To synchronize the current state of the robot with the trajectory tracker.
        set_ref_states: Set the local reference states for the coming time step.
        run_step: Run one step of trajectory tracking.
        check_termination_condition: Check if the robot finishes the trajectory tracking.

    Notes:
        The solver needs to be built before running the trajectory tracking. \n
        To run the tracker: 
            1. Load motion model and init states; 
            2. Set reference path and trajectory; 
            3. Run step.
    """


    def __init__(self, config: MpcConfiguration, robot_specification: CircularRobotSpecification, robot_id:Optional[int]=None, use_tcp:bool=False, verbose=False):
        """Initialize the trajectory tracker.

        Args:
            config: MPC configuration.
            robot_specification: Robot specification.
            use_tcp: If the PANOC solver is called via TCP or not. Defaults to False.
            verbose: If print out debug information. Defaults to False.
        """
        print(f"[{self.__class__.__name__}] Initializing robot {robot_id if robot_id is not None else 'X'}...")

        self.vb = verbose
        self.config = config
        self.robot_spec = robot_specification

        # Common used parameters from config
        self.ts = self.config.ts
        self.ns = self.config.ns
        self.nu = self.config.nu
        self.N_hor = self.config.N_hor
        self.solver_type = self.config.solver_type
        self.safe_zone_weight  = self.config.safe_zone_weight
        # Initialization
        self._idle = True
        self._map_loaded = False

        self._safe_zone_weight = [self.safe_zone_weight ] * self.N_hor
        self.set_work_mode(mode='safe')

        if self.config.solver_type == 'PANOC':
            self.__import_solver(use_tcp=use_tcp)

    def __import_solver(self, root_dir:str='', use_tcp:bool=False):
        """Import the PANOC solver from the build directory.

        Args:
            root_dir: The root directory of the PANOC solver. Defaults to ''.
            use_tcp: If the PANOC solver is called via TCP or not. Defaults to False.
        """

        self.use_tcp = use_tcp
        solver_path = os.path.join(root_dir, self.config.build_directory, self.config.optimizer_name)

        import opengen as og
        if not use_tcp:
            sys.path.append(solver_path)
            #built_solver = __import__(self.config.optimizer_name) # it loads a .so (shared library object)
            sys.path.insert(1, './mpc_grid_solver/grid_solver')
            import grid_solver
            #built_solver =
            #self.solver = built_solver.solver() # Return a Solver object with run method, cannot find it though
            self.solver = grid_solver.solver() # Return a Solver object with run method, cannot find it though

        else: # use TCP manager to access solver
            self.mng:og.opengen.tcp.OptimizerTcpManager = og.tcp.OptimizerTcpManager(solver_path)
            self.mng.start()
            self.mng.ping() # ensure RUST solver is up and runnings


    @property
    def idle(self):
        return self._idle
    


    #this is used in the set up process in the main like     controller.load_motion_model(UnicycleModel(sampling_time=config_mpc.ts))
    def load_motion_model(self, motion_model: Callable) -> None:
        """Load the motion model (dynamics) of the robot.

        Args:
            motion_model: The motion model of the robot, should be a Callable object.

        Notes:
            The motion model should be `s'=f(s,a,ts)` (takes in a state and an action and returns the next state).
        """
        self.motion_model = motion_model



    def load_init_states(self, current_state: np.ndarray, goal_state: np.ndarray):
        """Load the initial state and goal state.

        Args:
            current_state: Current state of the robot.
            goal_state: Final goal state of the robot (used to decelerate if close to goal).

        Raises:
            TypeError: All states should be numpy.ndarry.

        Attributes:
            state: Current state of the robot.
            final_goal: Goal state of the robot.
            past_states: List of past states of the robot.
            past_actions: List of past actions of the robot.
            cost_timelist: List of cost values of the robot.
            solver_time_timelist: List of solver time of the robot.
            finishing: If the robot is approaching the final goal.

        Notes:
            This function resets the `idx_ref_traj/path` to 0 and `idle` to False.
        """

        if (not isinstance(current_state, np.ndarray)) or (not isinstance(goal_state, np.ndarray)):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}/{type(goal_state)}.')
        self.state = current_state
        self.final_goal = goal_state

        self.past_states: List[np.ndarray] = []
        self.past_actions: List[np.ndarray] = []
        self.cost_timelist: List[float] = []
        self.solver_time_timelist: List[float] = []

        self._idle = False
        self.finishing = False # If approaching the last node of the reference path

    def set_work_mode(self, mode:str='safe'):
        """Set the basic work mode (base speed and weight parameters) of the MPC solver.

        Args:
            mode: Be "aligning" (start) or "safe" (20% speed) or "work" (80% speed) or "super" (full speed). Defaults to 'safe'.

        Raises:
            ModuleNotFoundError: If the mode is not supported.

        Attributes:
            base_speed: The reference speed
            tuning_params: Penalty parameters for MPC

        Notes:
            This method will overwrite the base speed.
        """
        
        if mode == 'aligning':
            self.base_speed = self.robot_spec.lin_vel_max*0.5
            self.tuning_params = [0.0] * self.config.nq
            self.tuning_params[2] = 100
        else:
            self.tuning_params = [self.config.qpos, self.config.qvel, self.config.qtheta, self.config.lin_vel_penalty, self.config.ang_vel_penalty,
                                  self.config.qpN, self.config.qthetaN, self.config.qrpd, self.config.lin_acc_penalty, self.config.ang_acc_penalty]
            if mode == 'safe':
                self.base_speed = self.robot_spec.lin_vel_max*0.2
            elif mode == 'work':
                self.base_speed = self.robot_spec.lin_vel_max*0.8
            elif mode == 'super':
                self.base_speed = self.robot_spec.lin_vel_max*1.0
            else:
                raise ModuleNotFoundError(f'There is no mode called {mode}.')

    def set_current_state(self, current_state: np.ndarray):
        """To synchronize the current state of the robot with the trajectory tracker.

        Args:
            current_state: The actually current state of the robot.

        Raises:
            TypeError: The state should be numpy.ndarry.
        """
        if not isinstance(current_state, np.ndarray):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}.')
        self.state = current_state



    #this should be done each time step in the main loop  controller.set_ref_states(ref_states, ref_speed=ref_speed)
    def set_ref_states(self, ref_states: np.ndarray, ref_speed:Optional[float]=None):
        """Set the local reference states for the coming time step.

        Args:
            ref_states: Local (within the horizon) reference states
            ref_speed: The reference speed. If None, use the default speed.
            
        Notes:
            This method will overwrite the base speed.
        """
        self.ref_states = ref_states
        if ref_speed is not None:
            self.base_speed = ref_speed


    #you need to probably use it differently to how it is being used now. 
    def check_termination_condition(self, external_check=True) -> bool:
        """Check if the robot finishes the trajectory tracking.

        Args:
            external_check: If this is true, the controller will check if it should terminate. Defaults to True.

        Returns:
            _idle: If the robot finishes the trajectory tracking.
        """
        if external_check:
            self.finishing = True
            if np.allclose(self.state[:2], self.final_goal[:2], atol=0.1, rtol=0) and abs(self.past_actions[-1][0]) < 0.1:
                self._idle = True
                if self.vb:
                    print(f"[{self.__class__.__name__}] Trajectory tracking finished.")
        return self._idle
    


    
    def run_step(self, ref_states : list[PathNode], static_obstacles:Optional[list]=None, full_dyn_obstacle_list:Optional[list]=None, other_robot_states:Optional[list]=None, map_updated:bool=True, path_tree_nodes_geo:list[ np.ndarray]=[]):
        """Run the trajectory planner for one step given the surrounding environment.

        Args:
            static_obstacles: A list of static obstacles, each element is a list of points (x,y).
            full_dyn_obstacle_list: A list of dynamic obstacles. Defaults to None.
            other_robot_states: A list of other robots' states. Defaults to None.
            map_updated: If the map is updated at this time step. Defaults to True.

        Returns:
            actions: A list of future actions
            pred_states: A list of predicted states
            ref_states: Reference states
            cost: The cost of the predicted trajectory
            closest_obstacle_list: A list of closest static obstacles
        """
        if map_updated or (not self._map_loaded):
            self._map_loaded = True
        closest_refpoints = self.closest_convex_hull_to_ref_point(reference_path=ref_states, path_tree_nodes_geo= path_tree_nodes_geo)
        convex_hull_constraints = self._halfspace_representation(closest_refpoints=closest_refpoints)
        actions, pred_states, ref_states, cost = self._run_step(convex_hull_constraints=convex_hull_constraints)
        return actions, pred_states, ref_states, cost


    def closest_convex_hull_to_ref_point(self, reference_path: list[PathNode], path_tree_nodes_geo: list[np.ndarray]=[]):
        """
        Determine the closest convex hull to each reference point in the given path.

        Args:
            reference_path (list): List of reference points (PathNodes) to which to find the closest convex hull.
            path_tree_nodes_geo (list): List of geometric representations of path tree nodes.

        Returns:
            closest_refpoints (dict): Dictionary containing the closest convex hulls to each reference point.

        Comments:
            This function iterates through each reference point in the given path and finds the closest convex hull
            to that point among the provided geometric representations of path tree nodes. It computes the distance
            between each reference point and the convex hulls using half-space representations and selects the one
            with the smallest distance. The result is stored in a dictionary with each reference point as the key
            and the corresponding closest convex hull as the value.
        """
        closest_refpoints:dict[PathNode, tuple[cs.SX, cs.SX, cs.SX]] = {}

        for ref_point in reference_path:
            x, y = ref_point[:2]
            state = cs.vcat([x,y,0.0]) # ref path
            cost_dict = {}

            for safe_area in path_tree_nodes_geo:
                b, a0, a1 = self.one_halfspace(safe_area)
                b_cs = cs.SX(b)
                a0_cs = cs.SX(a0)
                a1_cs = cs.SX(a1)
                outside_convex = self.outside_cvx_polygon(point=state, b=b_cs.T, a0=a0_cs.T, a1=a1_cs.T)
                cost_dict[outside_convex]=(b, a0, a1)

            closest_convexhull_to_point = min(cost_dict.items(), key=lambda item: item[0])
            closest_refpoints[(x,y)] = closest_convexhull_to_point[1] # taking the value from (outside_convex, (b, a0, a1) )

        return closest_refpoints




    def _halfspace_representation(self, closest_refpoints:dict[PathNode, tuple[cs.SX,cs.SX,cs.SX]]):
        num_params_per_convex_hull =12 #put this in config file
        convex_hull_constraints = [0.0] * (self.N_hor * num_params_per_convex_hull)  #len(convex_hull_list) 10 closest 
        for i , value in enumerate(closest_refpoints.values()):
            b, a0, a1 = value
            convex_hull_constraints[i * num_params_per_convex_hull : (i+1)*num_params_per_convex_hull] = b + a0 + a1
        #print("convex_hull_constraints", convex_hull_constraints)
        return convex_hull_constraints


    @staticmethod
    def one_halfspace(polygon_points: np.ndarray):
        hull = ConvexHull(polygon_points)
        hull_center = np.mean(polygon_points[hull.vertices, :], axis=0)  # (1xd) vector

        K = hull.simplices
        V = polygon_points - hull_center # perform affine transformation
        A_ = np.nan * np.empty((K.shape[0], polygon_points.shape[1]))

        rc = 0
        for i in range(K.shape[0]):
            ks = K[i, :]
            F = V[ks, :]
            if np.linalg.matrix_rank(F) == F.shape[0]:
                f = np.ones(F.shape[0])
                A_[rc, :] = np.linalg.solve(F, f)
                rc += 1

        A:np.ndarray = A_[:rc, :]
        b:np.ndarray = np.dot(A, hull_center.T) + 1.0
        return b.tolist(), A[:,0].tolist(), A[:,1].tolist()


    @staticmethod
    def outside_cvx_polygon(point: cs.SX, b: cs.SX, a0: cs.SX, a1: cs.SX) -> cs.SX:
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


    def _run_step(self, static_obstacles:Optional[list]=None, full_dyn_obstacle_list:Optional[list]=None, other_robot_states:Optional[list]=None, map_updated:bool=True, convex_hull_constraints:list=[]):
        """Run the trajectory planner for one step, wrapped by `run_step`.

        Args:
            other_robot_states: A list with length "ns*N_hor*Nother" (E.x. [0,0,0] * (self.N_hor*self.config.Nother)). Defaults to None.

        Raises:
            RuntimeError: If the solver cannot be run.

        Returns:
            actions: A list of future actions
            pred_states: A list of predicted states
            ref_states: Reference states
            cost: The cost of the predicted trajectory
        """

        ### Get reference states ###
        ref_states = self.ref_states.copy()
        finish_state = ref_states[-1,:]
        current_refs = ref_states.reshape(-1).tolist()

        ### Get reference velocities ###
        dist_to_goal = math.hypot(self.state[0]-self.final_goal[0], self.state[1]-self.final_goal[1]) # change ref speed if final goal close
        if (dist_to_goal < self.base_speed*self.N_hor*self.ts) and self.finishing:
            speed_ref = dist_to_goal / self.N_hor / self.ts
            speed_ref = min(speed_ref, self.robot_spec.lin_vel_max)
            speed_ref_list = [speed_ref]*self.N_hor
        else:
            speed_ref_list = [self.base_speed]*self.N_hor

        last_u = self.past_actions[-1] if len(self.past_actions) else np.zeros(self.nu)
            
        ### Assemble parameters for solver & Run MPC###
        #print("number of speed", speed_ref_list,len(speed_ref_list))
        #print("number of convex", len(convex_hull_constraints))

        params = list(last_u) + list(self.state) + list(finish_state) + self.tuning_params + \
                 current_refs + speed_ref_list + convex_hull_constraints + self._safe_zone_weight
        

        print("number of params", len(params))
        print("number of last_u", len(last_u))
        print("number of state", len(self.state))
        print("number of finish_state", len(finish_state))
        print("number of tuning_params", len(self.tuning_params))
        print("number of current_refs", len(current_refs))
        print("number of speed_ref_list", len(speed_ref_list))
        print("number of convex_hull_constraints", len(convex_hull_constraints))
        print("number of _safe_zone_weight", len(self._safe_zone_weight))

        try:
            # self.solver_debug(stc_constraints) # use to check (visualize) the environment
            taken_states, pred_states, actions, cost, solver_time, exit_status = self.run_solver(params, self.state, self.config.action_steps)
            if exit_status in self.config.bad_exit_codes and self.vb:
                print(f"[{self.__class__.__name__}] Bad converge status: {exit_status}")
        except RuntimeError:
            if self.use_tcp:
                self.mng.kill()
            raise RuntimeError(f"[{self.__class__.__name__}] Cannot run solver.")

        self.past_states.append(self.state)
        self.past_states += taken_states[:-1]
        self.past_actions += actions
        self.state = taken_states[-1]
        self.cost_timelist.append(cost)
        self.solver_time_timelist.append(solver_time)

        return actions, pred_states, ref_states, cost


    def run_solver(self, parameters:list, state: np.ndarray, take_steps:int=1):
        """Run the solver for the pre-defined MPC problem.

        Args:
            parameters: All parameters used by MPC, defined while building.
            state: The current state.
            take_steps: The number of control step taken by the input. Defaults to 1.

        Raises:
            ModuleNotFoundError: If the solver type is not supported.

        Returns:
            taken_states: List of taken states, length equal to take_steps.
            pred_states: List of predicted states at this step, length equal to horizon N.
            actions: List of taken actions, length equal to take_steps.
            cost: The cost value of this step
            solver_time: Time cost for solving MPC of the current time step
            exit_status: The exit state of the solver.

        Notes:
            The motion model (dynamics) is defined initially.
        """
        if self.solver_type == 'PANOC':
            if self.use_tcp:
                return self.run_solver_tcp(parameters, state, take_steps)

            import opengen as og
            solution:og.opengen.tcp.solver_status.SolverStatus = self.solver.run(parameters)
            print(solution)
            u:List[float]       = solution.solution
            cost:float          = solution.cost
            exit_status:str     = solution.exit_status
            solver_time:float   = solution.solve_time_ms

        else:
            raise ModuleNotFoundError(f'There is no solver with type {self.solver_type}.')
            
        taken_states:List[np.ndarray] = []
        for i in range(take_steps):
            state_next = self.motion_model(state, np.array(u[(i*self.nu):((i+1)*self.nu)]), self.ts)
            taken_states.append(state_next)

        pred_states:List[np.ndarray] = [taken_states[-1]]
        for i in range(len(u)//self.nu):
            pred_state_next = self.motion_model(pred_states[-1], np.array(u[(i*self.nu):(2+i*self.nu)]), self.ts)
            pred_states.append(pred_state_next)
        pred_states = pred_states[1:]

        actions = np.array(u[:self.nu*take_steps]).reshape(take_steps, self.nu).tolist()
        actions = [np.array(action) for action in actions]
        return taken_states, pred_states, actions, cost, solver_time, exit_status

    def run_solver_tcp(self, parameters:list, state: np.ndarray, take_steps:int=1):
        solution = self.mng.call(parameters)
        if solution.is_ok(): # Solver returned a solution
            solution = solution.get()
            u:List[float]       = solution.solution
            cost:float          = solution.cost
            exit_status:str     = solution.exit_status
            solver_time:float   = solution.solve_time_ms
        else: # Invocation failed - an error report is returned
            solver_error = solution.get()
            error_code = solver_error.code
            error_msg = solver_error.message
            self.mng.kill() # kill so rust code wont keep running if python crashes
            raise RuntimeError(f"[{self.__class__.__name__}] MPC Solver error: [{error_code}]{error_msg}")

        taken_states:List[np.ndarray] = []
        for i in range(take_steps):
            state_next = self.motion_model( state, np.array(u[(i*self.nu):((i+1)*self.nu)]), self.ts )
            taken_states.append(state_next)

        pred_states:List[np.ndarray] = [taken_states[-1]]
        for i in range(len(u)//self.nu):
            pred_state_next = self.motion_model( pred_states[-1], np.array(u[(i*self.nu):(2+i*self.nu)]), self.ts )
            pred_states.append(pred_state_next)
        pred_states = pred_states[1:]

        actions = np.array(u[:self.nu*take_steps]).reshape(take_steps, self.nu).tolist()
        actions = [np.array(action) for action in actions]
        return taken_states, pred_states, actions, cost, solver_time, exit_status
    
