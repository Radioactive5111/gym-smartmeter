import gymnasium as gym
# import gym_smartmeter
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
from stable_baselines3 import PPO
import numpy as np


import torch
# print(torch.cuda.is_available())
# if torch.cuda.is_available():
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)
# else:
#     print("CUDA is not available. Using CPU instead.")

def index_to_action(action_index):
    return action_index * 5000

def action_to_index(action):
    return action/5000

def compute_cost(load):
    non_peak_kwh = sum(load[0:419]) / 60000
    peak_kwh = sum(load[420:1440])/ 60000
    return (13.22 * non_peak_kwh + 30.41 * peak_kwh) / 100

df_loaded = pd.read_csv('D:\\Project\\smartmeter0\\UKData_1day_clean\\sum_all_1d.csv', index_col=0)

battery_capacity = 90000
total_steps = 1440
origin_values = df_loaded.loc['Total Usage'].tolist()
LAMBDA = 1

# Create environment
env = gym.make('smartmeter-v2',
               total_steps=total_steps,
               battery_capacity=battery_capacity,
               origin_values=origin_values,
               trade_off_factor=LAMBDA)

check_env(env, warn=True, skip_render_check=True)

# # Instantiate the agent
ppo_params = {
    'learning_rate': 5e-4,  # 学习率 3e-4
    'n_steps': 2048,  # 时间步长
    'batch_size': 32,  # 批量大小 64
    'clip_range': 0.2,  # 减少剪切范围
    'ent_coef': 0.1,  # 增加熵系数 0.1
    'vf_coef': 0.5,  # 值函数损失系数
    'max_grad_norm': 0.5  # 梯度裁剪
}
# # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_smartmeter_v0_tensorboard/",**ppo_params)
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_smartmeter_v0_tensorboard/")
# # # model = PPO.load("PPO_smartmeterbox_v0", env=env)
# #
# # # Train the agent and display a progress bar
# model.learn(total_timesteps=total_steps * 1000)
# # #
# # # Save the agent
# model.save("PPO_smartmeterbox_v1")
# del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO.load("PPO_smartmeterbox_v1", env=env)
#
# # Evaluate the agent
# # NOTE: If you use wrappers with your environment that modify rewards,
# #       this will be reflected here. To evaluate with original rewards,
# #       wrap environment in a "Monitor" wrapper before other wrappers.
# # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# # Enjoy trained agent
[obs, info] = env.reset()
print("Init Obs:", obs)
masked_values = []
actions = []
legal_actions = []
rewards = []
battery = []
for i in range(1440):
    # print("Time:", i)
    action, _ = model.predict(obs, deterministic=True)
    # action = np.random.uniform(-1, 1)
    # print("Action: ", index_to_action(action))

    obs, reward, done, truncated, info = env.step(action)
    # print('obs=', obs, 'reward=', reward, 'done=', done, 'info=', info)
    actions.append(index_to_action(action))
    legal_actions.append(info['Legal action'])
    battery.append(obs[1])
    rewards.append(reward)
    masked_values.append(info['Masked'])

sliding_avg = env.sliding_avg
origin_cost = compute_cost(origin_values)
masked_cost = compute_cost(masked_values)
print("Original cost:{} GBP, Masked cost:{} GBP, Cost:{} GBP".format(origin_cost, masked_cost, masked_cost-origin_cost))
print("Battery level:{}".format(battery[-1]))
# np.save('PPO_results.npy', masked_values)

def plot_actions(total_steps, actions, legal_actions):
    peaks = env.peak
    peak_indices = [i for i, number in enumerate(peaks) if number != 0]
    plt.plot(range(total_steps), actions, label='Actions', linewidth=5)
    plt.plot(range(total_steps), legal_actions, label='Legal Actions', linewidth=1)
    plt.scatter(
        [i for i in peak_indices],
        [actions[i] for i in peak_indices],
        color='red', marker='x', label='Peak'
    )
    plt.xlabel('Time Step')
    plt.ylabel('Action')
    plt.title('Action by PPO')
    plt.legend()
    plt.show()
    plt.close()
def plot_curve(total_steps, origin, mask):
    plt.plot(range(total_steps), origin, label='Original Value',linewidth=5)
    plt.plot(range(total_steps), mask, 'orange', label='Masked Value', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Origin Curve Value')
    plt.title('origin and masked PPO')
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.close()
def plot_SoC(total_steps,battery):
    peaks = env.peak
    peak_indices = [i for i, number in enumerate(peaks) if number != 0]
    plt.plot(range(total_steps),battery)
    plt.title('SoC by PPO')
    plt.scatter(
        [i for i in peak_indices],
        [battery[i] for i in peak_indices],
        color='red', marker='x', label='Peak'
    )
    plt.show()
    plt.close()
def plot_sliding(total_steps,sliding_avg, origin):
    plt.title('sliding_avg')
    plt.plot(range(total_steps), origin, label='Original Value',linewidth=5)
    plt.plot(range(total_steps), sliding_avg, 'orange', label='Sliding Mean', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Origin Curve Value')
    plt.title('Original Curve and Sliding Mean')
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.close()
# Plot
plot_actions(total_steps, actions, legal_actions)
plot_curve(total_steps, origin_values, masked_values)
plot_SoC(total_steps,battery)
# plot_sliding(total_steps,sliding_avg, load_level)



