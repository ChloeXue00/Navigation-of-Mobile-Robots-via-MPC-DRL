import random
import copy
import os
import pathlib

import gymnasium as gym
from torch import no_grad
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from utils.plotresults import plot_training_results
from utils.map import generate_map_dynamic, generate_map_corridor, generate_map_mpc
from pkg_ddpg_td3.environment.environment import TrajectoryPlannerEnvironment
from pkg_ddpg_td3.environment.agent import MobileRobot
from gymnasium.envs.registration import register
from pkg_ddpg_td3.environment.components import Component
# register(
#     id='TrajectoryPlannerEnvironment-v0',
#     entry_point='pkg_ddpg_td3.environment:TrajectoryPlannerEnvironment',  # 修改为你的实际路径
# )
# register(
#     id='TrajectoryPlannerEnvironmentRaysReward1-v0',
#     entry_point='pkg_ddpg_td3.environment.variants.rays_reward1:TrajectoryPlannerEnvironment',
#     kwargs={
#         "components": [Component()],  # 替换为实际组件实例
#         "generate_map": generate_map_dynamic,  
#         "time_step": 0.2,  
#     },
# )


TO_TRAIN = True
TO_SAVE = True

def generate_map():
    """
    Generates a random map for the environment.
    """
    return random.choice([generate_map_dynamic, generate_map_corridor, generate_map_mpc()])()

def run():
    # Selects which predefined agent model to use
    index = 1  # Change index to test different variants

    # Parameters for DDPG/TD3 models
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
        path = f'./././Model/image'
    elif index == 1:
        path = f'./././Model/ray'
    else:
        raise ValueError('Invalid index')

    # Define save paths
    # Define save paths
    base_dir = os.path.expanduser("~/DRL-Traj-Planner/Model/ddpg/training")
    os.makedirs(base_dir, exist_ok=True)  # 确保目录存在
    path = base_dir


    tot_timesteps = int(1e6)
    n_cpu = 4  # Adjust based on your system

    # Initialize environment
    env_eval = gym.make(variant['env_name'], generate_map=generate_map_dynamic)
    check_env(env_eval)

    vec_env = make_vec_env(
        variant['env_name'],
        n_envs=n_cpu,
        seed=0,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'generate_map': generate_map_dynamic}
    )
    vec_env_eval = make_vec_env(
        variant['env_name'],
        n_envs=1,
        seed=0,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'generate_map': generate_map_dynamic()}
    )

    # Choose DDPG or TD3
    Algorithm = TD3 if index == 0 else DDPG

    eval_callback = EvalCallback(
        vec_env_eval,
        best_model_save_path=path,
        log_path=path,
        eval_freq=max(tot_timesteps // 100, 1)
    )

    # Initialize model
    model = Algorithm(
        "MultiInputPolicy",
        vec_env,
        gamma=0.99,
        learning_rate=0.0001,
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        policy_kwargs={'net_arch': variant['net_arch']},
        device=variant['device'],
        verbose=1
    )

    # Train the model
    if TO_TRAIN:
        model.learn(total_timesteps=tot_timesteps, log_interval=4, callback=eval_callback)
        # Save the model
        if TO_SAVE:
            model.save(f"{path}/final_model")

    # Load the trained model
    model = Algorithm.load(f"{path}/best_model", env=vec_env)

    # Plot the learning curve of the training run
    plot_training_results(f"{path}/evaluations.npz")

    # Run the agent in an evaluation environment indefinitely
    env_eval_sim = gym.make(variant['env_name'], generate_map=generate_map)

    with no_grad():
        while True:
            obsv = env_eval.reset()
            for i in range(1000):
                action, _states = model.predict(obsv, deterministic=True)
                obsv, reward, done, info = env_eval.step(action)

                real_robot: MobileRobot = env_eval.agent
                pred_positions = []
                robot_sim = copy.deepcopy(real_robot)
                for j in range(20):
                    if j == 0:
                        robot_sim.step(action, 0.1)
                    else:
                        robot_sim.step_with_decay_angular_velocity(0.1, j+1)
                    pred_positions.append(list(robot_sim.position))

                if i % 3 == 0:
                    env_eval_sim.render(pred_positions=pred_positions)
                if done:
                    break

if __name__ == "__main__":
    run()
