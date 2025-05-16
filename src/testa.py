import os
import pathlib
import random
import copy
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from torch import no_grad
from stable_baselines3 import DDPG,TD3
from pkg_ddpg_td3.utils.per_ddpg import PerDDPG
from stable_baselines3.common import env_checker

from pkg_ddpg_td3.environment import MobileRobot
from pkg_ddpg_td3.environment.environment import TrajectoryPlannerEnvironment

from interface_mpc import InterfaceMpc
from util.mpc_config import Configurator

from helper_main_continous import generate_map, get_geometric_map, HintSwitcher, Metrics
from pkg_ddpg_td3.utils.map import test_scene_1_dict,test_scene_2_dict
from pkg_ddpg_td3.environment.components import Component
from timer import PieceTimer, LoopTimer
from typing import List, Tuple

import warnings
warnings.filterwarnings("ignore")

from gymnasium.envs.registration import register

register(
    id='TrajectoryPlannerEnvironmentRaysReward1-v0',
    entry_point='pkg_ddpg_td3.environment.variants.rays_reward1:TrajectoryPlannerEnvironment',
    kwargs={
        "Component":[Component()],
        "generate_map": generate_map,
        "time_step":0.2        
    }
)

MAX_RUN_STEP =200
DYN_OBS_SIZE = 0.8 + 0.8

def ref_traj_filter(original: np.array, new:np.ndarray, decay=1):
    filtered = original.copy()
    for i in range(filtered.shape[0]):
        filtered[i,:] = (1-decay)*filtered[i,:] + decay * new[i,:]
        decay *= decay
        if decay < 1e-2:
            decay=0
    return filter


def load_rl_model_env(generate_map,  index: int) -> Tuple[PerDDPG,TD3,TrajectoryPlannerEnvironment]:
    variant =[
        {
            'env_name':'TrajectoryPlannerEnvironmentImgsReward1-v0',
            'net_arch':[64,64],
            'per':False,
            'device':'auto'
        },
        {
           'env_name':'TrajectoryPlannerEnvironmentImgsReward1-v0',
           'net_arch': [16,16],
           'per':False,
           'device':'cpu' 
        }
    ][index]
    
    if index ==0:
        model_folder_name = 'image'
    elif index ==1:
        model_folder_name = 'ray'
    else:
        raise ValueError('Invalid index')
    model_path_ddpg = os.path.join(pathlib.Path(__file__).resolve().parents[1],'Model/ddpg', model_folder_name, 'best_model')
    model_path_td3 = os.path.join(pathlib.Path(__file__).resolve().parents[1],'Model/td3',model_folder_name)
    
    env_eval:TrajectoryPlannerEnvironment = gym.make(variant['env_name'], generate_map=generate_map)
    start_time = timer()
    env_checker.check_env(env_eval)
    print(f"Environment check time: {timer()-start_time}s") 
    start_time = timer()
    td3_model = TD3.load(model_path_td3)
    ddpg_model = PerDDPG.load(model_path_ddpg)
    print(f"Model loading time:{timer()-start_time}s")
    return ddpg_model,td3_model,env_eval

def load_mpc(config_path: str):
    config = Configurator(config_path)
    traj_gen = InterfaceMpc(config,motion_model=None)
    return traj_gen
    
def est_dyn_obs_position(last_pos:list,current_pos:list,steps:int=20):
    """
    estimate the dynamic obstacle position in the future
    """
    est_pos = []
    d_pos = [current_pos[0]-last_pos[0], current_pos[1]-last_pos[1]]
    for i in range(steps):
        est_pos.append([current_pos[0]+d_pos[0]*(i+1), current_pos[1]+d_pos[1]*(i+1),DYN_OBS_SIZE,DYN_OBS_SIZE,0,1])
    return est_pos


def circle_to_rect(pos: list, radius: float= DYN_OBS_SIZE):
    """
    convert the circle to a rectangle.
    """
    return[[pos[0]-radius, pos[1]-radius],[pos[0]+radius,pos[1]-radius],[pos[0]+radius,pos[1]+radius],[pos[0]-radius,pos[1]+radius]]

def main(rl_index:int=1,decision_mode:int=1, to_plot=False, scene_option:Tuple[int,int,int]=(1,1,1),save_num:int=1):
    """_summary_

    Args:
        rl_index (int, optional): _description_. Defaults to 1.
        decision_mode (int, optional): _description_. Defaults to 1.
        to_plot (bool, optional): _description_. Defaults to False.
        scene_option (Tuple[int,int,int], optional): _description_. Defaults to (1,1,1).
        save_num (int, optional): _description_. Defaults to 1.
    """
    prt_decision_mode = {0: 'pure_mpc', 1:'pure_ddpg', 2:'pure_td3',3:'hybrid_ddpg',4:'hybrid_td3'}
    print(f"The decision mode is:{prt_decision_mode[decision_mode]}")
    time_list = []
    
    ddpg_model, td3_model, env_eval = load_rl_model_env(generate_map(*scene_option),rl_index)
    env_eval = env_eval.unwrapped
    
    CONFIG_FN = 'mpc_longiter.yaml'
    cfg_fpath = os.path.join(pathlib.Path(__file__).resolve().parents[1],'config')
    traj_gen = load_mpc(cfg_fpath)
    geo_map= get_geometric_map(env_eval.get_map_description(),inflate_margin=0.8)
    traj_gen.update_static_constraints(geo_map.processed_obstacle_list)
    
    done =False
    with no_grad():
        while not done:
            obsv, *_ = env_eval.reset()
            
            init_state = np.array([*env_eval.agent.position,env_eval.agent.angle])
            goal_state = np.array([*env_eval.goal.position,0])
            ref_path = list(env_eval.path.coords)
            traj_gen.initialization(init_state,goal_state,ref_path)
            
            last_mpc_time = 0.0
            last_rl_time = 0.0
            
            chosen_ref_traj = None
            rl_ref = None
            last_dyn_obstacle_list = None
            switch = HintSwitcher(10,2,10)
            
            for i in range(0, MAX_RUN_STEP):
                
                dyn_obstacle_list = [obs.keyframe.position.tolist() for obs in env_eval.obstacles if not obs.is_static]
                dyn_obstacle_tmp = [obs+[DYN_OBS_SIZE,DYN_OBS_SIZE,0,1] for obs in dyn_obstacle_list] #扩展障碍物区域obs_with_size = [x, y, width, height, 0, 1]  # 包括尺寸信息

                dyn_obstacle_list_poly = [circle_to_rect(obs) for obs in dyn_obstacle_list] #把圆形障碍物变成多边形，polygon =[[x1,y1],[x2,y2],x3,y3],[x4,y4]
                dyn_obstacle_pred_list =[]
                if last_dyn_obstacle_list is None:
                    last_dyn_obstacle_list = dyn_obstacle_list
                for j, dyn_obs in enumerate(dyn_obstacle_list):
                    dyn_obstacle_pred_list.append(est_dyn_obs_position(last_pos=dyn_obstacle_list[j],current_pos=dyn_obs)) # 这里的dyn_obstacle_list[j] 和dyn_obs有什么不一样,因为index=j 好像是对应着enumerate dyn_obs
                last_dyn_obstacle_list = dyn_obstacle_list
                
                if decision_mode==0:
                    env_eval.set_agent_state(traj_gen.state[:2],traj_gen.state[2],
                                             traj_gen.last_action[0], traj_gen.last_action[1])  # state[:2]和state[2]是分别是位置[x,y]和方向吗，last action 是线速度和角速度吗
                    timer_rl = PieceTimer()
                    action_index,_states=td3_model.predict(obsv, deterministic=True)
                    obsv,reward,done,info = env_eval.step([0,0])
                    
                    if dyn_obstacle_list:
                        traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                    original_ref_traj,*_ = traj_gen.get_local_ref_traj()
                    chosen_ref_traj = original_ref_traj
                    timer_mpc = PieceTimer()
                    try:
                        mpc_output = traj_gen.get_action(chosen_ref_traj) #mpc_output=[v,ω] 或者=[v_x,v_y,ω]
                    except Exception as e:
                        done = True
                        print(f'MPC fails:{e}')
                        break
                    last_mpc_time = timer_mpc(4,ms =True)
                    if mpc_output is None:
                        break
                    action, pred_states,cost=mpc_output
                    
                elif decision_mode ==1:
                    traj_gen.set_current_state(env.eval.agent.state)
                    original_ref_traj, *_ = traj_gen.get_local_ref_traj()
                    
                    timer_rl = PieceTimer()
                    action_index,_states = ddpg_model.predict(obsv,deterministic=True)
                    last_rl_time = timer_rl(4, ms=True)
                    obsv, reward, terminated, truncated, info = env_eval.step(action_index)
                    
                    
                elif decision_mode ==2:
                    traj_gen.set_current_state(env_eval.agent.state)
                    original_ref_traj, *_ = traj_gen.get_local_ref_traj()
                    
                    timer_rl = PieceTimer()
                    action_index,_states = td3_model.predict(obsv,deterministic=True)
                    last_rl_time = timer_rl(4,ms=True)
                    obsv, reward, terminated, truncated, info = env_eval.step(action_index)
                
                elif decision_mode ==3:
                    env_eval.set_agent_state(traj_gen.state[:2],traj_gen.state[2],traj_gen.last_action[0],traj_gen.last_action[1])
                    timer_rl=PieceTimer()
                    action_index,_states =ddpg_model.predict(obsv,deterministic=True)
                    # obsv,reward,done,info = env_eval.step(action_index)
                    # # Manual step
                    # env_eval.step_obstacles()
                    # env_eval.update_status(reset=False)
                    # obsv = env_eval.get_observation()
                    # done = env_eval.update_termination()
                    # info= = env_eval.get_info()
                    
                    rl_ref = []
                    robot_sim:MobileRobot = copy.deepcopy(env_eval.agent)
                    robot_sim:MobileRobot
                    for j in range(20):
                        if j==0:
                            robot_sim.step(action_index, traj_gen.config.ts)
                        else:
                            robot_sim.step_with_ref_speed(traj_gen.config.ts,1.0)
                        rl_ref.append(list(robot_sim.position))
                    last_rl_time = timer_rl(4,ms=True)
                    
                    if dyn_obstacle_list:
                        traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                    original_ref_traj,rl_ref_traj,extra_ref_traj = traj_gen.get_local_ref_traj(np.array(rl_ref),20)
                    filtered_ref_traj = ref_traj_filter(original_ref_traj,rl_ref_traj,decay=1)
                    if switch.switch(traj_gen.state[:2],extra_ref_traj.tolist(),filtered_ref_traj.tolist(),geo_map.processed_obstacle_list):
                        chosen_ref_traj = filtered_ref_traj
                    else:
                        chosen_ref_traj = original_ref_traj
                    timer_mpc=PieceTimer()
                    try:
                        mpc_output = traj_gen.get_action(chosen_ref_traj)
                    except Exception as e:
                        done=True
                        print(f'MPC fails:{e}')
                        break
                    last_mpc_time = timer_mpc(4,ms=True)
                        
                elif decision_mode==4:
                    env_eval.set_agent_state(traj_gen.state[:2],traj_gen.state[2],
                                            traj_gen.last_action[0],traj_gen.last_action[1])
                    timer_rl =PieceTimer()
                    action_index, _states = td3_model.predict(obsv,deterministic=True)
                    env_eval.step_obstacles()
                    env_eval.update_status(reset=False)
                    obsv = env_eval.get_observation()
                    done = env_eval.update_termination()
                    info = env_eval.get_info()
                    
                    rl_ref = []
                    robot_sim:MobileRobot = copy.deepcopy(env_eval.agent)
                    robot_sim:MobileRobot
                    for j in range(20):
                        if j ==0:
                            robot_sim.step(action_index,traj_gen.config.ts)
                        else:
                            robot_sim.step_with_ref_speed(traj_gen.config.ts,1.0)
                        rl_ref.append(list(robot_sim.position))
                    last_rl_time = timer_rl(4,ms=True)
                    
                    if dyn_obstacle_list:
                        traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                    original_ref_traj,rl_ref_traj,extra_ref_traj = traj_gen.get_local_ref_traj(np.array(rl_ref))
                    filtered_ref_traj = ref_traj_filter(original_ref_traj,rl_ref_traj,decay=1)
                    if switch.switch(traj_gen.state[:2],original_ref_traj.tolist(),filtered_ref_traj.tolist(),geo_map.processed_obstacle_list+dyn_obstacle_list_poly):
                        chosen_ref_traj = filtered_ref_traj
                    else:
                        chosen_ref_traj = original_ref_traj
                    timer_mpc = PieceTimer()
                    try:
                        mpc_output=PieceTimer()
                    except Exception as e:
                        done = True
                        print(f'MPC fails:{e}')
                        break
                    last_mpc_time = timer_mpc(4,ms=True)
                    
                else:
                    raise ValueError("Invalid decision mode")
                if decision_mode ==0:
                    time_list.append(last_mpc_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (MPC):{last_mpc_time}ms")
                elif decision_mode == 1:
                    time_list.append(last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (DDPG): {last_rl_time}ms")
                elif decision_mode ==2:
                    time_list.append(last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (TD3): {last_rl_time}ms")
                elif decision_mode ==3:
                    time_list.append(last_mpc_time+last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (Hybrid DDPG):{last_mpc_time+last_rl_time}={last_mpc_time}+{last_rl_time}ms")
                elif decision_mode ==4:
                    time_list.append(last_mpc_time+last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (Hybrid TD3): {last_mpc_time+last_rl_time} = {last_mpc_time}+{last_rl_time}ms")
                        
                if to_plot &(i%1==0):
                    env_eval.render(dqn_ref = rl_ref, actual_ref=chosen_ref_traj,original_ref_traj=original_ref_traj,save =False,save_num = save_num )
                    
                if i==MAX_RUN_STEP -1:
                    done=True
                    print('Time out')
                if done:
                    if to_plot:
                        input(f"Finish (Succeed: {info['success']})! Press enter to continue...")
                    break
    print(f"Average time ({prt_decision_mode[decision_mode]}):{np.mean(time_list)}ms\n")
    return time_list

if __name__ == '__main__':
    """_summary_
    test_scene_1_dict = {1:[1,2,3],2:[1,2,3,4],3:[1,2,3,4],4:[1,2]}
    test_scene_2_dict = {1:[1,2,3]}
    
    decision_mode: 0 = MPC, 1=DDPG, 2=TD3, 3=Hybrid DDPG, 4=Hybrid TD3
    """
    """
    
    scene_option=(sub_index,scene_option)
    Subscene index ('sub_index') with scene option ('scene_option'):
    - 1: Single rectangular static obstacle 
        - (1-small, 2-medium, 3-large)
    - 2: Two rectangular static obstacles
        - (1-small stagger, 2- large stagger, 3-close aligned, 4-far aligned)
    - 3: Single non-convex static obstacle
        - (1-big u-shape, 2-small u-shape, 3-big v-shape, 4-small v-shape)
    - 4: Single dynamic obstacle
        - (1-crash, 2-cross) 
       
    """
    scene_option = (1,1,1)
    
    time_list_hyb_img = main(rl_index=0, decision_mode=3, to_plot=True, scene_option=scene_option, save_num=5)
    
    input('Press enter to exit...')
    
                
                        
                            
                    
                    
                    
                    
                    
                        