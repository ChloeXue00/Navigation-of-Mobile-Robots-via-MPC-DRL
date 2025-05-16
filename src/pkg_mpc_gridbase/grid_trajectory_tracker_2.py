#NOT CURRENTLY USED IN THE FINAL VERSION

# System import
import os
import sys
import math
import itertools
from typing import Callable, List, Optional, Type
# External import
import numpy as np
from scipy.spatial import ConvexHull # type: ignore
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

        # Initialization
        self._idle = True
        self._map_loaded = False
        self._safe_zone_weight = [100] * self.N_hor
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
    
    def extract_half_space_params_list(self, convex_hull_list):
            num_params_per_convex_hull =12
            convex_hull_constraints = [0.0] * (10 * num_params_per_convex_hull)  #len(convex_hull_list) 10 closest 
            print("convex inside",len(convex_hull_constraints))

            for i, convex_hull in enumerate(convex_hull_list):
                if i < 10:
                    vertices = convex_hull.points  # Get the vertices of the convex hull
                    edges = convex_hull.simplices  # Get the edges of the convex hull

                    b_list = []
                    a0_list = []
                    a1_list = []
                    for edge in edges:
                        v1, v2 = vertices[edge[0]], vertices[edge[1]]
                        normal = np.array([v2[1] - v1[1], v1[0] - v2[0]])  # Compute the normal vector to the edge

                        # Normalize the normal vector
                        normal /= np.linalg.norm(normal)

                        # Compute the parameters of the half-space inequality b - [a0, a1] * [x, y]' > 0
                        b = np.dot(normal, v1)
                        a0 = -normal[0]
                        a1 = -normal[1]

                        b_list.append(b)
                        a0_list.append(a0)
                        a1_list.append(a1)
                    convex_hull_constraints[i * num_params_per_convex_hull : (i+1)*num_params_per_convex_hull] = b_list + a0_list + a1_list
            print("convex inside",len(convex_hull_constraints))
            return convex_hull_constraints

    def run_step(self, static_obstacles:Optional[list]=None, full_dyn_obstacle_list:Optional[list]=None, other_robot_states:Optional[list]=None, map_updated:bool=True, path_tree_nodes_geo:dict[tuple,ConvexHull]={}):
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
        convex_hull_constraints= self.extract_half_space_params_list(convex_hull_list=path_tree_nodes_geo.values())
        #print("number of convex before", len(convex_hull_constraints),convex_hull_constraints )
        actions, pred_states, ref_states, cost = self._run_step(convex_hull_constraints=convex_hull_constraints)
        return actions, pred_states, ref_states, cost


        #given a list of convex hulls take each convex hull and break it into sub convex hull element representation ai bi a1i
        #put these element in a list a new list which stores all ai bi for ALL CONVEX HULLS 

 
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
        #print("number of params", len(params))
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
    
