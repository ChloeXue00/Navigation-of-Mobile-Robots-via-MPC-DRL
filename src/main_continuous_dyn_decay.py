##代码的核心是测试 MPC、DDPG、TD3 
##加载训练好的模型，验证其在不同环境和决策逻辑下的性能。
# 不更新模型参数，只评估现有策略。
### System import
import os
import pathlib
import random
import copy
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt

### DRL import
import gymnasium as gym
from torch import no_grad
from stable_baselines3 import DDPG,TD3
from pkg_ddpg_td3.utils.per_ddpg import PerDDPG
from stable_baselines3.common import env_checker

from pkg_ddpg_td3.environment import MobileRobot
from pkg_ddpg_td3.environment.environment import TrajectoryPlannerEnvironment

### MPC import
from interface_mpc import InterfaceMpc
from util.mpc_config import Configurator

### Helper
from helper_main_continous import generate_map, get_geometric_map, HintSwitcher, Metrics
from pkg_ddpg_td3.utils.map import test_scene_1_dict, test_scene_2_dict
#from pkg_dqn.utils.map import test_scene_1_dict, test_scene_2_dict
from pkg_ddpg_td3.environment.components import Component
### Others
from timer import PieceTimer, LoopTimer
from typing import List, Tuple

import warnings
warnings.filterwarnings("ignore")

#add register
from gymnasium.envs.registration import register

#import obstacle
from pkg_ddpg_td3.environment.obstacle import Obstacle

from shapely.geometry import Point


register(
    id='TrajectoryPlannerEnvironmentRaysReward1-v0',
    entry_point='pkg_ddpg_td3.environment.variants.rays_reward1:TrajectoryPlannerEnvironment',
    kwargs={
        "components": [Component()],  # 替换为实际组件实例
        "generate_map": generate_map,  
        "time_step": 0.2,  
    },
)


MAX_RUN_STEP = 200
DYN_OBS_SIZE = 0.8 + 0.8

#dyn decay
def adjust_decay(
    robot: MobileRobot,
    obstacles: List[Obstacle],
    safe_distance: float = 2.0,
    critical_distance: float = 1.0,
    max_decay: float = 1.0,
    min_decay: float = 0.1
) -> float:
    """
    自适应调整 decay 值。

    Args:
        robot (MobileRobot): 当前的机器人对象。
        obstacles (List[Obstacle]): 周围障碍物列表。
        safe_distance (float): 安全距离，超过这个距离时 decay 达到最小值。
        critical_distance (float): 临界距离，小于这个距离时 decay 达到最大值。
        max_decay (float): decay 的最大值。
        min_decay (float): decay 的最小值。

    Returns:
        float: 调整后的 decay 值。
    """
    # 初始化最小距离为一个较大的值
    min_distance = float('inf')

    # 遍历障碍物，计算机器人到障碍物的最短距离
    for i in range(len(obstacles)):
        for obstacle in obstacles[i]:
            distance = obstacle.padded_polygon.distance(robot.point)
            min_distance = min(min_distance, distance)

    # 根据距离动态调整 decay
    if min_distance <= critical_distance:
        # 非常接近障碍物，设置为最大衰减
        return max_decay
    elif min_distance >= safe_distance:
        # 远离障碍物，设置为最小衰减
        return min_decay
    else:
        # 根据距离线性插值调整 decay
        normalized_distance = (min_distance - critical_distance) / (safe_distance - critical_distance)
        return min_decay + (max_decay - min_decay) * (1 - normalized_distance)

##add ref_speed based on decay
def compute_ref_speed(self, robot: MobileRobot, obstacles: List[Obstacle], default_speed: float = 1.0, critical_distance: float = 1.0) -> float:
    """
    动态调整参考速度。
    """
    min_distance = min([obstacle.padded_polygon.distance(robot.point) for obstacle in obstacles])
    
    if min_distance < critical_distance:
        return default_speed * 0.5  # 临近障碍物时减速
    else:
        return default_speed  # 正常速度

def step_with_decay_angular_velocity(self, time_step: float, ref_speed: float = None, obstacles: List[Obstacle] = None) -> None:
    # 动态计算衰减系数和参考速度
    decay = 0.95
    if obstacles:
        decay = self.compute_decay(self, obstacles)
        ref_speed = self.compute_ref_speed(self, obstacles, default_speed=ref_speed)

    # 动态调整角速度
    self.angular_velocity *= decay
    self.angle += time_step * self.angular_velocity

    # 动态调整参考速度
    if ref_speed is None:
        ref_speed = self.speed
    elif ref_speed <= 0.0:
        ref_speed = self.cfg.SPEED_MAX

    # 更新位置
    self.position += time_step * ref_speed * np.asarray((np.cos(self.angle), np.sin(self.angle)))


def ref_traj_filter(original: np.ndarray, new: np.ndarray, decay: float):
    filtered = original.copy()
    for i in range(filtered.shape[0]):
        filtered[i, :] = (1-decay) * filtered[i, :] + decay * new[i, :]
        decay *= decay
        if decay < 1e-2:
            decay = 0.0
    return filtered

# #updated ref_tra_filter
# def ref_traj_filter(
#     original: np.ndarray,
#     new: np.ndarray,
#     robot: MobileRobot,
#     obstacles: List[Obstacle],
#     decay=1.0,
#     safe_distance=2.0,
#     critical_distance=1.0
# ):
#     """
#     Filter reference trajectory based on adaptive decay.

#     Args:
#         original: Original reference trajectory.
#         new: New reference trajectory.
#         robot: MobileRobot instance.
#         obstacles: List of obstacles.
#         decay: Initial decay value.
#         safe_distance: Safe distance threshold for obstacles.
#         critical_distance: Critical distance threshold for obstacles.

#     Returns:
#         Filtered trajectory.
#     """
#     filtered = original.copy()
#     for i in range(filtered.shape[0]):
#         decay = adjust_decay(decay, robot, obstacles, safe_distance, critical_distance)
#         filtered[i, :] = (1 - decay) * filtered[i, :] + decay * new[i, :]

#     return filtered


def load_rl_model_env(generate_map, index: int) -> Tuple[PerDDPG, TD3, TrajectoryPlannerEnvironment]:
    variant = [
        {
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward1-v0',
            'net_arch': [64, 64],
            'per': False,
            'device': 'auto',
        },
        {
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
            'net_arch': [16, 16],
            'per': False,
            'device': 'cpu',
        },
    ][index] 

    if index == 0:
        model_folder_name = 'image'
    elif index == 1:
        model_folder_name = 'ray'
    else:
        raise ValueError('Invalid index')
    model_path_ddpg = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'Model/ddpg', model_folder_name, 'best_model')
    model_path_td3 = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'Model/td3', model_folder_name, 'best_model')
    
    env_eval:TrajectoryPlannerEnvironment = gym.make(variant['env_name'], generate_map=generate_map)
    start_time = timer()
    env_checker.check_env(env_eval)
    print(f"Environment check time: {timer()-start_time}s")
    start_time = timer()
    td3_model = TD3.load(model_path_td3)
    ddpg_model = PerDDPG.load(model_path_ddpg)
    print(f"Model loading time: {timer()-start_time}s")
    return ddpg_model, td3_model, env_eval

def load_mpc(config_path: str):
    config = Configurator(config_path)
    traj_gen = InterfaceMpc(config, motion_model=None) # default motion model is used
    return traj_gen

def est_dyn_obs_positions(last_pos: list, current_pos: list, steps:int=20):
    """
    Estimate the dynamic obstacle positions in the future.
    """
    est_pos = []
    d_pos = [current_pos[0]-last_pos[0], current_pos[1]-last_pos[1]]
    for i in range(steps):
        est_pos.append([current_pos[0]+d_pos[0]*(i+1), current_pos[1]+d_pos[1]*(i+1), DYN_OBS_SIZE, DYN_OBS_SIZE, 0, 1])
    return est_pos

def circle_to_rect(pos: list, radius:float=DYN_OBS_SIZE):
    """
    Convert the circle to a rectangle.
    """
    return [[pos[0]-radius, pos[1]-radius], [pos[0]+radius, pos[1]-radius], [pos[0]+radius, pos[1]+radius], [pos[0]-radius, pos[1]+radius]]


def main(rl_index:int=1, decision_mode:int=1, to_plot=False, scene_option:Tuple[int, int, int]=(1, 1, 1), save_num:int=1):
    """
    Args:
        rl_index: 0 for image, 1 for ray
        decision_mode: 0 for pure rl, 1 for pure mpc, 2 for hybrid
    """
    prt_decision_mode = {0: 'pure_mpc', 1: 'pure_ddpg', 2: 'pure_td3', 3: 'hybrid_ddpg', 4: 'hybrid_td3'}
    print(f"The decision mode is: {prt_decision_mode[decision_mode]}")

    time_list = []

    ddpg_model, td3_model, env_eval = load_rl_model_env(generate_map(*scene_option), rl_index)
    env_eval = env_eval.unwrapped

    CONFIG_FN = 'mpc_longiter.yaml'
    cfg_fpath = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', CONFIG_FN)
    traj_gen = load_mpc(cfg_fpath)
    geo_map = get_geometric_map(env_eval.get_map_description(), inflate_margin=0.8)
    traj_gen.update_static_constraints(geo_map.processed_obstacle_list) # if assuming static obstacles not changed

    done = False
    with no_grad():
        while not done:
            obsv, *_ = env_eval.reset()

            init_state = np.array([*env_eval.agent.position, env_eval.agent.angle])
            goal_state = np.array([*env_eval.goal.position, 0])
            ref_path = list(env_eval.path.coords)
            #print(ref_path)
            traj_gen.initialization(init_state, goal_state, ref_path)

            last_mpc_time = 0.0
            last_rl_time = 0.0

            chosen_ref_traj = None
            rl_ref = None
            last_dyn_obstacle_list = None      

            switch = HintSwitcher(10, 2, 10)

            for i in range(0, MAX_RUN_STEP):
                
                dyn_obstacle_list = [obs.keyframe.position.tolist() for obs in env_eval.obstacles if not obs.is_static]
                dyn_obstacle_tmp  = [obs+[DYN_OBS_SIZE, DYN_OBS_SIZE, 0, 1] for obs in dyn_obstacle_list]
                dyn_obstacle_list_poly = [circle_to_rect(obs) for obs in dyn_obstacle_list]

                dyn_obstacle_pred_list = []
                if last_dyn_obstacle_list is None:
                    last_dyn_obstacle_list = dyn_obstacle_list
                for j, dyn_obs in enumerate(dyn_obstacle_list):
                    dyn_obstacle_pred_list.append(est_dyn_obs_positions(last_dyn_obstacle_list[j], dyn_obs))
                last_dyn_obstacle_list = dyn_obstacle_list
                
                
                if decision_mode == 0:
                    env_eval.set_agent_state(traj_gen.state[:2], traj_gen.state[2], 
                                                traj_gen.last_action[0], traj_gen.last_action[1])
                    obsv, reward, terminated, truncated, info = env_eval.step([0,0]) # just for plotting and updating status

                    if dyn_obstacle_list:
                        traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                    original_ref_traj, *_ = traj_gen.get_local_ref_traj()
                    chosen_ref_traj = original_ref_traj
                    timer_mpc = PieceTimer()
                    try:
                        mpc_output = traj_gen.get_action(chosen_ref_traj)
                    except Exception as e:
                        done = True
                        print(f'MPC fails: {e}')
                        break
                    last_mpc_time = timer_mpc(4, ms=True)
                    if mpc_output is None:
                        break
                    action, pred_states, cost = mpc_output

                elif decision_mode == 1:
                    traj_gen.set_current_state(env_eval.agent.state)
                    original_ref_traj, *_ = traj_gen.get_local_ref_traj() # just for output

                    timer_rl = PieceTimer()
                    action_index, _states = ddpg_model.predict(obsv, deterministic=True)
                    last_rl_time = timer_rl(4, ms=True)
                    obsv, reward, terminated, truncated, info = env_eval.step(action_index)
                    ### Manual step
                    # env_eval.step_obstacles()
                    # env_eval.update_status(reset=False)
                    # obsv = env_eval.get_observation()
                    # done = env_eval.update_termination()
                    # info = env_eval.get_info()

                elif decision_mode == 2:
                    traj_gen.set_current_state(env_eval.agent.state)
                    original_ref_traj, *_ = traj_gen.get_local_ref_traj() # just for output

                    timer_rl = PieceTimer()
                    action_index, _states = td3_model.predict(obsv, deterministic=True)
                    last_rl_time = timer_rl(4, ms=True)
                    obsv, reward, terminated, truncated, info = env_eval.step(action_index)

                elif decision_mode == 3:
                    env_eval.set_agent_state(traj_gen.state[:2], traj_gen.state[2], 
                                             traj_gen.last_action[0], traj_gen.last_action[1])
                    timer_rl = PieceTimer()
                    action_index, _states = ddpg_model.predict(obsv, deterministic=True)
                    # obsv, reward, done, info = env_eval.step(action_index)
                    ### Manual step
                    env_eval.step_obstacles()
                    env_eval.update_status(reset=False)
                    obsv = env_eval.get_observation()
                    done = env_eval.update_termination()
                    info = env_eval.get_info()

                    rl_ref = []
                    robot_sim:MobileRobot = copy.deepcopy(env_eval.agent)
                    robot_sim:MobileRobot
                    
                    
                    for j in range(20):
                        if j == 0:
                            robot_sim.step(action_index, traj_gen.config.ts)
                        else:
                            robot_sim.step_with_ref_speed(traj_gen.config.ts, 1.0)
                        rl_ref.append(list(robot_sim.position))
                    last_rl_time = timer_rl(4, ms=True)
                    # last_rl_ref = rl_ref
                    
                    if dyn_obstacle_list:
                        # traj_gen.update_dynamic_constraints([dyn_obstacle_tmp*20])
                        traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                    ##add a dyn decay
                    # decay = adjust_decay(
                    #     robot=robot_sim,
                    #     obstacles=dyn_obstacle_list_poly + geo_map.processed_obstacle_list,  # 传入动态和静态障碍物列表
                    #     safe_distance=2.0,
                    #     critical_distance=1.0,
                    #     max_decay=1.0,
                    #     min_decay=0.1
                    # )
                    original_ref_traj, rl_ref_traj, extra_ref_traj = traj_gen.get_local_ref_traj(np.array(rl_ref),20)
                    filtered_ref_traj = ref_traj_filter(original_ref_traj, rl_ref_traj, decay=0.5) # decay=1 means no decay
                    # filtered_ref_traj = ref_traj_filter(original_ref_traj, rl_ref_traj, decay=decay) # decay=decay means using the dyn decay 

                    if switch.switch(traj_gen.state[:2], extra_ref_traj.tolist(), filtered_ref_traj.tolist(), geo_map.processed_obstacle_list+dyn_obstacle_list_poly):
                        chosen_ref_traj = filtered_ref_traj
                    else:
                        chosen_ref_traj = original_ref_traj
                    timer_mpc = PieceTimer()
                    try:
                        mpc_output = traj_gen.get_action(chosen_ref_traj) # MPC computes the action
                    except Exception as e:
                        done = True
                        print(f'MPC fails: {e}')
                        break
                    last_mpc_time = timer_mpc(4, ms=True)

                elif decision_mode == 4:
                    env_eval.set_agent_state(traj_gen.state[:2], traj_gen.state[2], 
                                             traj_gen.last_action[0], traj_gen.last_action[1])
                    timer_rl = PieceTimer()
                    action_index, _states = td3_model.predict(obsv, deterministic=True)
                    # obsv, reward, done, info = env_eval.step(action_index)
                    ### Manual step
                    env_eval.step_obstacles()
                    env_eval.update_status(reset=False)
                    obsv = env_eval.get_observation()
                    done = env_eval.update_termination()
                    info = env_eval.get_info()

                    rl_ref = []
                    robot_sim:MobileRobot = copy.deepcopy(env_eval.agent)
                    robot_sim:MobileRobot
                    for j in range(20):
                        # if j == 0:
                        robot_sim.step(action_index, traj_gen.config.ts)
                        # else:
                            # robot_sim.step_with_ref_speed(traj_gen.config.ts, 1.0)
                        rl_ref.append(list(robot_sim.position))
                    last_rl_time = timer_rl(4, ms=True)
                    # last_rl_ref = rl_ref
                    
                    if dyn_obstacle_list:
                        # traj_gen.update_dynamic_constraints([dyn_obstacle_tmp*20])
                        traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                    original_ref_traj, rl_ref_traj, extra_ref_traj = traj_gen.get_local_ref_traj(np.array(rl_ref))
                    filtered_ref_traj = ref_traj_filter(original_ref_traj, rl_ref_traj, decay=1) # decay=1 means no decay
                    if switch.switch(traj_gen.state[:2], original_ref_traj.tolist(), filtered_ref_traj.tolist(), geo_map.processed_obstacle_list+dyn_obstacle_list_poly):
                        chosen_ref_traj = filtered_ref_traj
                    else:
                        chosen_ref_traj = original_ref_traj
                    timer_mpc = PieceTimer()
                    try:
                        mpc_output = traj_gen.get_action(chosen_ref_traj) # MPC computes the action
                    except Exception as e:
                        done = True
                        print(f'MPC fails: {e}')
                        break
                    last_mpc_time = timer_mpc(4, ms=True)

                else:
                    raise ValueError("Invalid decision mode")
                
                if decision_mode == 0:
                    time_list.append(last_mpc_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (MPC): {last_mpc_time}ms")
                elif decision_mode == 1:
                    time_list.append(last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (DDPG): {last_rl_time}ms")
                elif decision_mode == 2:
                    time_list.append(last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (TD3): {last_rl_time}ms")
                elif decision_mode == 3:
                    time_list.append(last_mpc_time+last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (Hybrid DDPG): {last_mpc_time+last_rl_time} = {last_mpc_time}+{last_rl_time}ms")
                elif decision_mode == 4:
                    time_list.append(last_mpc_time+last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (Hybrid TD3): {last_mpc_time+last_rl_time} = {last_mpc_time}+{last_rl_time}ms") 


                if to_plot & (i%1==0): # render
                    env_eval.render(dqn_ref=rl_ref, actual_ref=chosen_ref_traj, original_ref=original_ref_traj, save=False, save_num=save_num)

                if i == MAX_RUN_STEP - 1:
                    done = True
                    print('Time out!')
                if done:
                    if to_plot:
                        input(f"Finish (Succeed: {info['success']})! Press enter to continue...")
                    break

    print(f"Average time ({prt_decision_mode[decision_mode]}): {np.mean(time_list)}ms\n")
    return time_list



if __name__ == '__main__':
    """
    test_scene_1_dict = {1: [1, 2, 3], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2]}
    test_scene_2_dict = {1: [1, 2, 3]}


    decision_mode: 0 = MPC, 1 = DDPG, 2 = TD3, 3 = Hybrid DDPG, 4 = Hybrid TD3  
    """
    scene_option = (1, 2, 3)

    # time_list_mpc     = main(rl_index=1,    decision_mode=0,  to_plot=True, scene_option=scene_option, save_num=1) # Eval MPC using main.py
    # time_list_img     = main(rl_index=0,    decision_mode=1,  to_plot=False, scene_option=scene_option, save_num=3) #use ddpg
    time_list_hyb_img = main(rl_index=0,    decision_mode=3,  to_plot=True, scene_option=scene_option, save_num=5)  #hybrid

    input('Press enter to exit...')