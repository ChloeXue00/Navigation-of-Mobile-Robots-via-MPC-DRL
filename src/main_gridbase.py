import os
import json
import pathlib
from typing import Any

import numpy as np
import matplotlib.pyplot as plt # type: ignore
from matplotlib.lines import Line2D # type: ignore
from matplotlib.patches import Rectangle # type: ignore

from basic_motion_model.motion_model import UnicycleModel
from pkg_motion_plan.local_traj_plan import LocalTrajPlanner
from pkg_robot.robot import Robot
from basic_obstacle.geometry_plain import PlainPolygon

from configs import MpcConfiguration
from configs import CircularRobotSpecification

from visualizer.object import PointSpeedVehicleVisualizer

from pkg_mpc_gridbase.grid_trajectory_tracker import TrajectoryTracker
from basic_map.map_geometric import GeometricMap
from map_processor import MapProcessor
from connective_quadtree import QuadTree

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#BUILD THE SOLVER FIRST BEFORE USING IT. RUN BUILD_SOLVER_G
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------



### Universal parameters
distance_threshold = 100



def inflate_map( original_map: GeometricMap, inflation_margin: float) -> GeometricMap:
    boundary_coords, obstacle_coords_list = original_map()
    for i, obs in enumerate(obstacle_coords_list):
        inflated_obs = PlainPolygon.from_list_of_tuples(obs).inflate(inflation_margin)
        obstacle_coords_list[i] = inflated_obs()
    boundary_polygon = PlainPolygon.from_list_of_tuples(boundary_coords).inflate(-inflation_margin)
    boundary_coords = boundary_polygon()
    return GeometricMap.from_raw(boundary_coords, obstacle_coords_list)

def dyn_get_occupancy_map( boundary, obstacle_list, rescale:float=100.0) -> np.ndarray:
    """
    Arguments:
        rescale: The resolution of the occupancy map (rescale * real size).
    Returns:
        A numpy array of shape (height, width, 3).
    """

    boundary_np = np.array(boundary)
    width  = max(boundary_np[:,0]) - min(boundary_np[:,0])
    height = max(boundary_np[:,1]) - min(boundary_np[:,1])

    fig, ax = plt.subplots(figsize=(width, height), dpi=rescale)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.plot(np.array(boundary)[:,0], 
            np.array(boundary)[:,1], 'w-')
    for obs in obstacle_list:
        x = np.array(obs)[:,0]
        y = np.array(obs)[:,1]
        plt.fill(x, y, color='k')
        print(x)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    occupancy_map = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    occupancy_map = occupancy_map.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    np.save("occupancy_map.npy", occupancy_map)
    print("Occupancy map saved as 'occupancy_map.npy' ")

    return occupancy_map

def main(case_index:int=0):
    """
    case_index: 
        0: Static obstacles
        1: Sharp turns
        2: Dynamic obstacles
    """
    boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    if case_index == 0: # Static obstacles
        robot_starts= np.asarray([2.0, 8.0, 0.0]) #innitial position 
        path_coords = [
            (1.0, 1.0), (3.0, 1.0), (4.0, 1.0), (7.0, 1.0), (8.2, 1.0), 
            (8.5, 2.0), (8.0, 6.0), (9.5, 8.0), (8.5, 9.0), (2.0, 9.0)
        ]
        path_coords = path_coords[::-1]
        obstacle_list = [
            {"id_": 1, "vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]},
            {"id_": 2, "vertices": [(5, 3), (6, 3), (6, 6), (5, 6), (8,8)]},
            {"id_": 3, "vertices": [(5, 1), (6, 1), (6, 6), (5, 6)]},
            {"id_": 4, "vertices": [(7, 3.5), (8, 3.5), (7, 4.5), (8, 4.5)]},
        ]
    elif case_index == 1: # Sharp turns
        robot_starts= np.asarray([1.0, 1.0, 0.0])
        path_coords = [
            (1.0, 1.0), (3.0, 1.0), (4.0, 1.0), (7.0, 1.0), (8.0, 3.0),
            (8.0, 6.0), (9.0, 8.0), (8.0, 9.0), (2.0, 8.0)
        ] 
        obstacle_list = [
            {"id_": 1, "vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]},
            {"id_": 2, "vertices": [(5, 3), (6, 3), (6, 6), (5, 6), (8,8)]},
            {"id_": 3, "vertices": [(5, 1), (6, 1), (6, 6), (5, 6)]}
        ]
    elif case_index == 2: # Dynamic obstacles
        robot_starts= np.asarray([1.5, 1.0, 0.0])
        path_coords = [
            (1.0, 1.0), (3.0, 1.0), (4.0, 1.0), (7.0, 1.0), (8.0, 3.0),
            (8.0, 6.0), (9.0, 8.0), (8.8, 9.0), (2.0, 9.0)
        ]
        obstacle_list = [
            {"id_": 1, "vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]},
            {"id_": 2, "vertices": [(5, 3), (6, 3), (6, 6), (5, 6), (8,8)]},
            {"id_": 3, "vertices": [(5, 1.1), (6, 1.1), (6, 6), (5, 6)]},
        ]
        dynamic_paths = [
            (1.0, 2.0), (1.0, 2.5), (1.0, 3.0), (1.5, 3.5), (2.0, 4.0), 
            (2.5, 4.5), (3.3, 4.8), (4.0, 5.0), (4.0, 5.5), (4.0, 6.0),
            (4.0, 6.5), (4.0, 7.0), (4.5, 7.5), (5.0, 8.0), (5.5, 8.5),
            (6.0, 9.0), (6.5, 9.0)
        ] 
    elif case_index == 3: # Rectangle
        robot_starts= np.asarray([1.0, 1.0, 0.0])
        path_coords = [
            (1.0, 1.0), (3.0, 1.0), (4.0, 1.0), (7.0, 1.0)]
        obstacle_list = [
            {"id_": 1, "vertices": [(3, 0.8), (4, 0.8), (4, 1.1), (3, 1.1)]},

        ]
        dynamic_paths = [
        ]       
    else:
        raise ValueError(f"Invalid case index: {case_index}")
    
    # Load files
    ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
    DATA_DIR = os.path.join(ROOT_DIR, "data", "schedule_demo1_data")
    CNFG_DIR = os.path.join(ROOT_DIR, "config")
    VB = False
    TIMEOUT = 1000

    # Configurations
    config_mpc = MpcConfiguration.from_yaml(os.path.join(CNFG_DIR, "mpc_default.yaml"))
    config_robot = CircularRobotSpecification.from_yaml(os.path.join(CNFG_DIR, "robot_spec.yaml"))

    # Load and inflate map
    obstacle_info_list = [GeometricMap.dict_to_obstacle_info(obs) for obs in obstacle_list]
    geometric_map = GeometricMap(boundary, obstacle_info_list)
    x_min, x_max, y_min, y_max = geometric_map.get_boundary_scope()

    processor = MapProcessor()
    processor.load_map(geometric_map, config_robot.vehicle_width)
    inflated_map = processor._inflated_map
    height, width, _ = inflated_map.shape
    rescale_geo_inflated_occupancy = height/(y_max-y_min)

    # Generate quadtree
    inflated_map_grayscale = np.dot(inflated_map[..., :3], [0.2989, 0.5870, 0.1140]) / 255.0
    inflated_map_binary = np.where(inflated_map_grayscale > 0.5, 0.0, 1.0)
    rescale_geo_occupancy = 100  # You need to define the appropriate value for your case
    quadtree = QuadTree(value_matrix=inflated_map_binary, min_cell_size=int(config_robot.vehicle_width*100), rescale_geo_occupancy=rescale_geo_occupancy)

    # Create the robot, local planner and controller
    robot = Robot(config_robot, UnicycleModel(sampling_time=config_mpc.ts),1)
    robot.set_state(np.asarray(robot_starts))

    planner = LocalTrajPlanner(config_mpc.ts, config_mpc.N_hor, config_robot.lin_vel_max, verbose=VB)
    planner.load_path(path_coords, path_times=None, nomial_speed=robot.config.lin_vel_max, method="linear")

    controller = TrajectoryTracker(config_mpc, config_robot, robot_id=1, verbose=VB)
    controller.load_motion_model(UnicycleModel(sampling_time=config_mpc.ts))
    # important ponts and expected states from the reference path which the controller needs to know about
    goal_coord = path_coords[-1]
    goal_coord_prev = path_coords[-2]
    goal_heading = np.arctan2(goal_coord[1]-goal_coord_prev[1], goal_coord[0]-goal_coord_prev[0])
    goal_state = np.array([*goal_coord, goal_heading])
    controller.load_init_states(robot_starts, goal_state)

    if case_index == 2: # create the square for dynamic obstacle
        square_size = 0.5
        square_center = dynamic_paths[0]
        half_size = square_size / 2

        top_left = (square_center[0] - half_size, square_center[1] + half_size)
        top_right = (square_center[0] + half_size, square_center[1] + half_size)
        bottom_left = (square_center[0] - half_size, square_center[1] - half_size)
        bottom_right = (square_center[0] + half_size, square_center[1] - half_size)
        vertices1 = [top_left, top_right, bottom_right, bottom_left]
        dyn_obstacle_list = [vertices1]
        dyn_np_array = dyn_get_occupancy_map(boundary, dyn_obstacle_list, rescale=rescale_geo_inflated_occupancy)
        dyn_occ = np.array(dyn_np_array)

        result2 = dyn_occ + inflated_map
        condition1= (result2 == [255,255,255])
        arr1 = np.where(condition1, [0,0,0], result2)
        condition2= (arr1 == [254,254,254])
        dyn_static_occupancy = np.where(condition2, [255,255,255], arr1)
        quadtree.update_Quadtree(new_occupancyMap= dyn_static_occupancy)

    ### START THE PLOTS
    viser = PointSpeedVehicleVisualizer(0.5, 1, 2)
    fig, ax = plt.subplots()
    ax.imshow(inflated_map, extent=[x_min, x_max, y_min, y_max])
    viser.plot(ax, *robot.state)

    # Plot the ref path
    ref_path_xs, ref_path_ys = zip(*path_coords)
    ax.plot(ref_path_xs, ref_path_ys, color='blue', marker='o')
    ref_plot = ax.plot([], [], color='red', marker='o', alpha=0.2)[0]

    hull_plots:list[Line2D] = []
    if case_index == 2:
        dyn_path_xs, dyn_path_ys = zip(*dynamic_paths)
        ax.plot(dyn_path_xs, dyn_path_ys, color='green', marker='o')
        square = Rectangle(bottom_left,square_size,square_size)
        square_patch = ax.add_patch(square)

    ### Running
    for kt in range(TIMEOUT):
        if case_index == 2:
            square_patch.remove()
            index = min(int(kt//5.5), len(dynamic_paths)-1)
            square_center = dynamic_paths[index]
            top_left = (square_center[0] - half_size, square_center[1] + half_size)
            top_right = (square_center[0] + half_size, square_center[1] + half_size)
            bottom_left = (square_center[0] - half_size, square_center[1] - half_size)
            bottom_right = (square_center[0] + half_size, square_center[1] - half_size)
            
            vertices1 = [top_left, top_right, bottom_right, bottom_left]
            dyn_obstacle_list = [vertices1]
            dyn_np_array = dyn_get_occupancy_map(boundary, dyn_obstacle_list,rescale=rescale_geo_inflated_occupancy )
            dyn_occ = np.array(dyn_np_array)
        
            #inflated map + dynamic ob map
            result2 = dyn_occ + inflated_map
            condition1= (result2 == [255,255,255])
            arr1 = np.where(condition1, [0,0,0], result2)
            condition2= (arr1 == [254,254,254])
            dyn_static_occupancy = np.where(condition2, [255,255,255], arr1)

            quadtree.update_Quadtree(new_occupancyMap= dyn_static_occupancy)
            square = Rectangle(bottom_left,square_size,square_size)
            square_patch = ax.add_patch(square)
        
        robot_states = []
        ref_plot.set_data([], [])  # Clear the previous references plots
        ref_states, ref_speed, *_ = planner.get_local_ref(kt*config_mpc.ts, (float(robot.state[0]), float(robot.state[1])) )
        
        
        #check the Local Global path 
        print(f"Local path reference points at step {kt}: {ref_states}")
        print(f"Global goal state: {goal_state}")

        #save the occupancy map
        np.save(f"dynamic_occupancy_map_step_{kt}.npy", dyn_static_occupancy)
        plt.imsave(f"dynamic_occupancy_map_step_{kt}.png", dyn_static_occupancy)


        controller.set_ref_states(ref_states, ref_speed=ref_speed)

        ref_traj_xs = [row[0] for row in ref_states]
        ref_traj_ys = [row[1] for row in ref_states]
        ref_plot.set_data(ref_traj_xs, ref_traj_ys)
        

        path_tree_nodes, path_tree_nodes_geo = quadtree.collect_tree_nodes_on_path(ref_states, height_occupancy=height, threshold=distance_threshold, rescale_geo_occupancy=rescale_geo_occupancy, rescale_geo_inflated_occupancy=rescale_geo_inflated_occupancy)

        # Clear the previous hull plots
        for hp in hull_plots:
            hp.remove()
        hull_plots.clear()

        # Plot the convex hulls in the geometric map
        for hull, points in path_tree_nodes_geo.values():
            for simplex in hull.simplices:
                hull_plot = ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], color='orange')[0]
                hull_plots.append(hull_plot)

        # Run controller
        path_tree_nodes_points_list = []
        [path_tree_nodes_points_list.append(points) for hull, points in path_tree_nodes_geo.values()] # type: ignore
        actions, pred_states, current_refs, cost = controller.run_step(ref_states= ref_states,map_updated=False, path_tree_nodes_geo= path_tree_nodes_points_list)
        print(f"Cost {cost}")

        # if cost > 5000:
        #     robot.step([0, 0]) # step the robot
        # else:
        print(len(actions), actions)
        
        # print(f"Step {kt}: Dynamic obstacle center: {square_center}")
    
        robot.step(actions[-1]) # step the robot
        if controller.check_termination_condition(external_check=planner.idle):
            break         
        robot_states.append(robot.state)
        viser.update(*robot.state)
        plt.pause(0.06)
    plt.show()


if __name__ == "__main__":
    main(case_index=2)
