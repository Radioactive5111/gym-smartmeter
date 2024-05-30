import gymnasium as gym
import gym_smartmeter
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure
import pandas as pd
from stable_baselines3 import DQN
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from replay_buffer import ReplayBuffer

# 定义 H-network
class HNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(HNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 双向 LSTM 输出大小是 hidden_size * 2

    def forward(self, x):
        h0 = torch.zeros(2 * num_layers, x.size(0), hidden_size).to(device)  # 初始化隐藏状态
        c0 = torch.zeros(2 * num_layers, x.size(0), hidden_size).to(device)  # 初始化细胞状态
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

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
check_env(env, warn=True, skip_render_check=True)

# 超参数设置
input_size = 1  # 每个时间步的数据是一个标量
hidden_size = 44  # 隐藏层大小
num_layers = 2  # LSTM 层数
output_size = 1  # 输出大小，预测当前时刻的原始负载
learning_rate_h = 0.001
learning_rate_q = 0.00025
num_episodes = 1000  # 训练周期数
# batch_size = 64 # 小批次大小
# buffer_size = 500  # 重放缓冲区大小
batch_size = 5  # 小批次大小
buffer_size = 10  # 重放缓冲区大小

# k = 500  # target network 更新频率
k = 100
k_prime = 8  # Q-network 更新频率

# 初始化 H-network、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
h_network = HNetwork(input_size, hidden_size, num_layers, output_size).to(device)
criterion_h = nn.CrossEntropyLoss()
optimizer_h = optim.RMSprop(h_network.parameters(), lr=learning_rate_h)

# Initialize model
policy_kwargs = dict(
    net_arch=[64, 64],  # 两个隐藏层，每层64个神经元
    activation_fn=nn.ReLU
)
# 设置日志记录器
log_dir = "./logs/"
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000, learning_rate=learning_rate_q,
            batch_size=128, gamma=1, policy_kwargs=policy_kwargs, target_update_interval=k)
# 使用 RMSProp 优化器
model.optimizer = optim.RMSprop(model.policy.parameters(), lr=0.00025)
model.set_logger(new_logger)

# 初始化第二个经验重放缓冲区
replay_buffer_II = ReplayBuffer(buffer_size=buffer_size)

# 计算隐私泄露信号
def calculate_privacy_leakage_signal(y_t, y_t_minus_1, z_T, h_network):
    # 将 y_t_minus_1 和 z_T 转换为 tensor
    y_t_minus_1 = torch.tensor(y_t_minus_1, dtype=torch.float32).unsqueeze(-1).to(device)
    z_T = torch.tensor(np.hstack(z_T), dtype=torch.float32).unsqueeze(-1).to(device)  # 展平 z_T

    # 拼接 y_t_minus_1 和 z_T
    x = torch.cat((y_t_minus_1, z_T), dim=0).unsqueeze(0)

    # 通过 H-network 计算条件概率 P(Y_t | Y^{t-1}, Z^T)
    with torch.no_grad():
        y_pred = h_network(x)

    # 计算隐私泄露信号
    privacy_leakage_signal = -torch.log(y_pred.squeeze()).item()

    return privacy_leakage_signal

# Main training loop
for episode in range(num_episodes):
    state,info = env.reset()
    masked_values = []
    loss_h = torch.tensor(0.0)
    for t in range(total_steps):
        # 使用 ε-greedy 策略选择动作
        action, _ = model.predict(state, deterministic=False)
        action = np.array([action])

        # 执行动作，获取新的状态和奖励
        next_state, _, done, truncated, info = env.step(action)
        masked_values.append(info['Masked'])

        # 计算隐私泄露信号
        if t > 0:  # 确保 y_t_minus_1 存在
            privacy_signal = calculate_privacy_leakage_signal(
                origin_values[t], origin_values[:t], masked_values, h_network
            )
        else:
            privacy_signal = 0  # 第一步时没有隐私泄露信号

        reward = - privacy_signal
        # 将经验存储到重放缓冲区 I
        model.replay_buffer.add(state, next_state, action, reward, done, [info])

        # 更新状态
        state = next_state

        # 定期更新 Q-network
        if t % k_prime == 0 and len(replay_buffer_II) >= batch_size:
            model.train(batch_size=batch_size, gradient_steps=1)

        # 定期更新 target-network 和 H-network
        if t % k == 0 and t >= 1:
            # model.update_target_network()
            # 更新 H-network
            if len(replay_buffer_II) >= batch_size:
                for _ in range(len(replay_buffer_II) // batch_size):
                    masked_vals, original_vals = replay_buffer_II.sample(batch_size)
                    # t_step = min(t, len(original_vals[0]))  # Ensure t_step is within bounds
                    x_batch = []
                    y_batch = []
                    x_batch = np.concatenate((np.array(original_vals)[:,:t],np.array(masked_vals)),axis=1)
                    y_batch = np.array(original_vals)[:,t]
                    x_batch = torch.nn.utils.rnn.pad_sequence(
                        [torch.tensor(seq, dtype=torch.float32).unsqueeze(-1) for seq in x_batch], batch_first=True).to(device)
                    y_batch = torch.tensor(np.array(y_batch), dtype=torch.float32).to(device).unsqueeze(-1)
                    h_network.train()
                    optimizer_h.zero_grad()
                    y_pred = h_network(x_batch)
                    # print(y_pred,y_batch)
                    loss_h = criterion_h(y_pred, y_batch)
                    # loss_h = nn.MSELoss(y_pred, y_batch)
                    print(loss_h)
                    loss_h.backward()
                    optimizer_h.step()

    # if episode==1:
    #     for _ in range(buffer_size):
    #         replay_buffer_II.store_transition(masked_values, origin_values)
    replay_buffer_II.store_transition(np.array(masked_values), np.array(origin_values))
    if len(replay_buffer_II) > buffer_size:
        replay_buffer_II.pop(0)


    # 评估当前策略
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(
        f'Episode {episode + 1}, H-network Loss: {loss_h.item():.4f}, DQN Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')

print("Training complete.")
# 保存模型
model.save("DDQLMI_v0")
torch.save(h_network.state_dict(), "h_network_smartmeter.pth")

# 清理环境
env.close()




# # Instantiate the agent
# model = DQN("MultiInputPolicy", env, verbose=1)
# model = DQN.load("DQN_MI_v1", env=env)

# Train the agent and display a progress bar
# model.learn(total_timesteps=total_steps * 1000)
#
# Save the agent
# model.save("DQN_MI_v2")
# del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
# model = DQN.load("DQN_MI_v2", env=env)
# model = DQN.load("DQN_smartmeter_v1", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)




# # Enjoy trained agent
# [obs, info] = env.reset()
# print("Init Obs:", obs)
#
# masked_values = []
# actions = []
# legal_actions = []
# load_level = df_loaded.loc['Total Usage'].tolist()
# rewards = []
# battery = []
#
# for i in range(1440):
#     # print("Time:", i)
#     action, _ = model.predict(obs, deterministic=True)
#     # print("Action: ", index_to_action(action))
#
#     obs, reward, done, truncated, info = env.step(action)
#     # print('obs=', obs, 'reward=', reward, 'done=', done, 'info=', info)
#     actions.append(index_to_action(action))
#     legal_actions.append(info['Legal action'])
#     battery.append(obs[1])
#     rewards.append(reward)
#     masked_values.append(info['Masked'])
#
# sliding_avg = env.sliding_avg
# origin_cost = compute_cost(load_level)
# masked_cost = compute_cost(masked_values)
# print("Original cost:{} GBP, Masked cost:{} GBP, Cost:{} GBP".format(origin_cost, masked_cost, masked_cost-origin_cost))
#
#
# # np.save('DDQL-MI_results.npy', masked_values)
#
# def plot_actions(total_steps, actions, legal_actions):
#     plt.plot(range(total_steps), actions, label='Actions', linewidth=5)
#     plt.plot(range(total_steps), legal_actions, label='Legal Actions', linewidth=1)
#     plt.xlabel('Time Step')
#     plt.ylabel('Action')
#     plt.title('Action by DQN')
#     plt.legend()
#     plt.show()
#     plt.close()
# def plot_curve(total_steps, origin, mask):
#     plt.plot(range(total_steps), origin, label='Original Value',linewidth=5)
#     plt.plot(range(total_steps), mask, 'orange', label='Masked Value', linewidth=1)
#     plt.xlabel('Time Step')
#     plt.ylabel('Origin Curve Value')
#     plt.title('origin and masked DQN')
#     plt.tight_layout()
#     plt.legend()
#     plt.show()
#     plt.close()
# def plot_SoC(total_steps,battery):
#     peaks = env.peak
#     peak_indices = [i for i, number in enumerate(peaks) if number != 0]
#     plt.plot(range(total_steps),battery)
#     plt.title('SoC by DQN')
#     plt.scatter(
#         [i for i in peak_indices],
#         [battery[i] for i in peak_indices],
#         color='red', marker='x', label='Peak'
#     )
#     plt.show()
#     plt.close()
# def plot_sliding(total_steps,sliding_avg, origin):
#
#     plt.title('sliding_avg')
#     plt.plot(range(total_steps), origin, label='Original Value',linewidth=5)
#     plt.plot(range(total_steps), sliding_avg, 'orange', label='Sliding Mean', linewidth=1)
#     plt.xlabel('Time Step')
#     plt.ylabel('Origin Curve Value')
#     plt.title('Original Curve and Sliding Mean')
#     plt.tight_layout()
#     plt.legend()
#     plt.show()
#     plt.close()
# # Plot
# plot_actions(total_steps, actions, legal_actions)
# plot_curve(total_steps, load_level, masked_values)
# plot_SoC(total_steps,battery)
# plot_sliding(total_steps,sliding_avg, load_level)

