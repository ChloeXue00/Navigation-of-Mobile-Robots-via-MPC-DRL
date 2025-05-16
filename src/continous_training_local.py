"""
Code used to train the continous DRL agents, DDPG and TD3.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

import sys
import gymnasium as gym
import numpy as np
import random
from gymnasium import Env

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from utils.plotresults import plot_training_results
# from stable_baselines3.common.env_checker import check_env
from gymnasium.utils.env_checker import check_env

from torch import no_grad
from pkg_ddpg_td3.utils.map import generate_map_dynamic, generate_map_corridor, generate_map_mpc, generate_map_eval
from pkg_ddpg_td3.environment import MapDescription
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from typing import Callable
from pkg_ddpg_td3.utils.per_ddpg import PerDDPG
from pkg_ddpg_td3.utils.per_td3 import PerTD3
from pkg_ddpg_td3.environment.components import Component
from pkg_ddpg_td3.environment.environment import TrajectoryPlannerEnvironment
from pkg_dqn.environment.environment import TrajectoryPlannerEnvironment
from pkg_ddpg_td3.environment.variants.imgs_reward1 import TrajectoryPlannerEnvironmentImgsReward1
# from pkg_ddpg_td3.environment.variants.imgs_reward2 import TrajectoryPlannerEnvironmentImgsReward2
from pkg_ddpg_td3.environment.variants.rays_reward1 import TrajectoryPlannerEnvironmentRaysReward1
from pkg_ddpg_td3.environment.variants.rays_reward2 import TrajectoryPlannerEnvironmentRaysReward2
import warnings
#add register
from gymnasium.envs.registration import register
warnings.filterwarnings("ignore")

from gymnasium.utils.env_checker import check_env



def generate_map() -> MapDescription:
    return random.choice([generate_map_dynamic, generate_map_corridor, generate_map_mpc()])()

register(
    id='TrajectoryPlannerEnvironmentRaysReward1-v0',
    entry_point='pkg_ddpg_td3.environment.variants.rays_reward1:TrajectoryPlannerEnvironment',
    kwargs={
        "components": [Component()],  # 替换为实际组件实例
        "generate_map": generate_map,  
        "time_step": 0.2,  
    },
)
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
    current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def run():
    
    # Selects which model variant to use
    # index = 6                   
    index = 2           

    # Load a pre-trained model
    # load_checkpoint = False
    load_checkpoint = True


    # Select the path where the model should be stored
    # path = f'./Model/local_training/variant-{index}'
    # path = '/home/fengxiang/DRL-Traj-Planner/Model/td3/image'
    path = '/home/fengxiang/DRL-Traj-Planner/Model/ddpg/ray/training'
    # path = './Model/ddpg/image'
    # path = '/home/fengxiang/DRL-Traj-Planner/Model/ddpg/ray'
    
    # Parameters for different example agent models 
    variant = [
        {
            'algorithm' : "DDPG",
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward1-v0',
            'net_arch': [64, 64],
            'per': True,
            'device': 'auto',
            'generate_map': generate_map_dynamic
        },
        {
            'algorithm' : "DDPG",
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward2-v0',
            'net_arch': [64, 64],
            'per': True,
            'device': 'auto',
            'generate_map': generate_map_dynamic
        },
        {
            'algorithm' : "DDPG",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
            'net_arch': [16, 16],
            'per': True,
            'device': 'cpu',
            'generate_map': generate_map_dynamic
        },
        {
            'algorithm' : "DDPG",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward2-v0',
            'net_arch': [16, 16],
            'per': True,
            'device': 'cpu',
            'generate_map': generate_map_dynamic
        },
        # TD3
        {
            'algorithm' : "TD3",
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward1-v0',
            'net_arch': [64, 64],
            'per': True,
            'device': 'auto',
        },
        {
            'algorithm' : "TD3",
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward2-v0',
            'net_arch': [64, 64],
            'per': True,
            'device': 'auto',
        },
        {
            'algorithm' : "TD3",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
            'net_arch': [16, 16],
            'per': True,
            'device': 'cpu',
        },
        {
            'algorithm' : "TD3",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward2-v0',
            'net_arch': [16, 16],
            'per': True,
            'device': 'cpu',
        },
    ][index]

    tot_timesteps = 10e4
    n_cpu = 12
    
    env_eval = gym.make(variant['env_name'], generate_map=generate_map_eval)
    
    #     # 定义子环境的工厂函数
    # def make_single_env():
    #     return gym.make(variant['env_name'], generate_map=generate_map_eval)
    # print('variant_env_name',variant['env_name'])
    # # 检查单个子环境
    # single_env = make_single_env()
    # observation, info = single_env.reset()
    # print(observation)
    # check_env(single_env)  # 检查单一环境是否符合 Gymnasium 规范
    
    
    # 定义并行环境工厂函数
    # def make_env(env_id, rank, generate_map):
    #     """
    #     创建一个特定环境实例的函数，用于 SubprocVecEnv。

    #     :param env_id: 环境名称字符串
    #     :param rank: 当前环境索引
    #     :param generate_map: 每个环境的地图生成函数
    #     """
    #     def _init():
    #         env = gym.make(env_id, generate_map=generate_map)
    #         env.reset(seed=rank)  # 在 reset 中设置随机种子
    #         print(f"环境 {rank} 已重置，种子设置为 {rank}")
    #         return env
    #     return _init
    # 2. 定义不同环境名称和地图生成函数
    # variant = [
        
    #     {
    #         'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
    #         'generate_map': generate_map_dynamic
    #     },
        
    #     {
    #         'env_name': 'TrajectoryPlannerEnvironmentRaysReward2-v0',
    #         'generate_map': generate_map_eval
    #     },
    # ]

    # 创建 SubprocVecEnv
    n_cpu = 4  # 使用的并行环境数量
    # env_id = 'TrajectoryPlannerEnvironmentImgsReward1-v0'
    # env_list = [make_env(variant[i % len(variant)]['env_name'], i, variant[i % len(variant)]['generate_map']) for i in range(n_cpu)]
    # vec_env = SubprocVecEnv(env_list)
    # print('2')

    # # vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(n_cpu),env_id in variant['env_name']])
    vec_env = make_vec_env(variant['env_name'], n_envs=n_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'generate_map': generate_map})
    vec_env_eval = make_vec_env(variant['env_name'], n_envs=n_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'generate_map': generate_map})
    # index = 0  # 假设选择第 0 个环境
    # env_config = variant[index]

    # vec_env_eval = make_vec_env(
    #     env_config['env_name'],
    #     n_envs=n_cpu,
    #     seed=0,
    #     vec_env_cls=SubprocVecEnv,
    #     env_kwargs={'generate_map': env_config['generate_map']}
    # )
    # print('3')

    # check_env(vec_env)

    n_actions  = vec_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions),sigma=0.1 * np.ones(n_actions))

    if variant["algorithm"] == "DDPG" and not variant["per"]:
        Algorithm = DDPG
    elif variant["algorithm"] == "DDPG" and variant["per"]:
        Algorithm = PerDDPG
    elif variant["algorithm"] == "TD3" and not variant["per"]:
        Algorithm = TD3
    elif variant["algorithm"] == "TD3" and variant["per"]:
        Algorithm = PerTD3

    
    eval_callback = EvalCallback(vec_env_eval,
                                 best_model_save_path=path,
                                 log_path=path,
                                 eval_freq=max((tot_timesteps / 100) // n_cpu, 1),
                                 n_eval_episodes=n_cpu)
    # print("当前环境 observation_space:", env_eval.observation_space)
    # import torch
    # model_data = torch.load("/home/fengxiang/DRL-Traj-Planner/Model/ddpg/ray/best_model.zip", map_location="cpu")
    # print("模型保存时的 observation_space:", model_data['observation_space'])

    if load_checkpoint:
        model = Algorithm.load(f"{path}/best_model", env=env_eval)
        plot_training_results(path)

        with no_grad():
            while True:
                obs = env_eval.reset()
                for i in range(0, 1000):
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env_eval.step(action)
                    if i % 3 == 0: # Only render every third frame for performance (matplotlib is slow)
                        # vec_env.render("human")
                        env_eval.render()
                    if done:
                        break
    
    
    else:
        model = Algorithm("MultiInputPolicy",
                    vec_env, 
                    # learning_rate=linear_schedule(0.0001),
                    learning_rate=0.0001, 
                    buffer_size=int(1e6), 
                    learning_starts=100_000, gamma=0.98,
                    gradient_steps=-1,
                    action_noise = action_noise,
                    policy_kwargs={'net_arch': variant['net_arch']},
                    verbose=1,
                    device=variant['device'],
                )

        # Train the model    
        model.learn(total_timesteps=tot_timesteps, log_interval=4, progress_bar=True, callback=eval_callback)


        # Save the model
        model.save(f"{path}/final_model")

                    
    
if __name__ == "__main__":
    run()
