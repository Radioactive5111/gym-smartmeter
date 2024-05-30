import os
import gymnasium as gym
import gym_smartmeter
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
from stable_baselines3 import DQN
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback



import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    print("CUDA is not available. Using CPU instead.")

def index_to_action(action_index):
    action = 20 * (action_index - 250)
    return action

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
env = gym.make('smartmeter-v1',
               total_steps=total_steps,
               battery_capacity=battery_capacity,
               origin_values=origin_values,
               trade_off_factor=LAMBDA)
eval_env = gym.make('smartmeter-v1',
               total_steps=total_steps,
               battery_capacity=battery_capacity,
               origin_values=origin_values,
               trade_off_factor=LAMBDA)

check_env(env, warn=True, skip_render_check=True)

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=1440,  # 评估频率
    deterministic=True,
    render=False
)
# Instantiate the agent
# model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_smartmeter_v1_tensorboard/")
# model = DQN("MlpPolicy", env, verbose=1)
model = DQN.load("DQN_smartmeterbox_v2.1", env=env)
# # #
# # # # Train the agent and display a progress bar
model.learn(total_timesteps=total_steps * 1000, callback=eval_callback)
# # # # #
# # # # Save the agent
# model.save("DQN_smartmeterbox_v2.2")
# del model  # delete trained model to demonstrate loading
#
# # Load the trained agent
# model = DQN.load('./logs/best_model', env=env)
best_model_path = os.path.join(log_dir, 'best_model.zip')
best_model = DQN.load(best_model_path, env=env)
# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
[obs, info] = env.reset()
print("Init Obs:", obs)

masked_values = []
actions = []
legal_actions = []
load_level = df_loaded.loc['Total Usage'].tolist()
rewards = []
battery = []

for i in range(1440):
    # print("Time:", i)
    action, _ = best_model.predict(obs, deterministic=True)
    # print("Action: ", index_to_action(action))

    obs, reward, done, truncated, info = env.step(action)
    # print('obs=', obs, 'reward=', reward, 'done=', done, 'info=', info)
    actions.append(index_to_action(action))
    legal_actions.append(info['Legal action'])
    battery.append(obs[1])
    rewards.append(reward)
    masked_values.append(info['Masked'])

sliding_avg = env.sliding_avg
origin_cost = compute_cost(load_level)
masked_cost = compute_cost(masked_values)
print("Original cost:{} GBP, Masked cost:{} GBP, Cost:{} GBP".format(origin_cost, masked_cost, masked_cost-origin_cost))
# np.save('DQN_v0.npy', masked_values)
print("final battery level", battery[1439])

def plot_actions(total_steps, actions, legal_actions):
    plt.plot(range(total_steps), actions, label='Actions', linewidth=5)
    plt.plot(range(total_steps), legal_actions, label='Legal Actions', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Action')
    plt.title('Action by DQN')
    plt.legend()
    plt.show()
    plt.close()

# plot curve

def plot_curve(total_steps, origin, mask):
    plt.plot(range(total_steps), origin, label='Original Value',linewidth=3)
    plt.plot(range(total_steps), mask, 'orange', label='Masked Value', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Origin Curve Value')
    plt.title('origin and masked DQN')
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.close()

def plot_SoC(total_steps,battery):
    peaks = env.peak
    peak_indices = [i for i, number in enumerate(peaks) if number != 0]
    plt.plot(range(total_steps),battery)
    plt.title('SoC by DQN')
    plt.scatter(
        [i for i in peak_indices],
        [battery[i] for i in peak_indices],
        color='red', marker='x', label='Peak'
    )
    plt.show()
    plt.close()

# Plot sliding_avg
def plot_sliding(total_steps,sliding_avg, origin):

    plt.title('sliding_avg')
    plt.plot(range(total_steps), origin, label='Original Value',linewidth=3)
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
plot_curve(total_steps, load_level, masked_values)
plot_SoC(total_steps,battery)
# plot_sliding(total_steps,sliding_avg, load_level)

