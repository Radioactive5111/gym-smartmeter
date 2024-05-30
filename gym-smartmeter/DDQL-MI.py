import gymnasium as gym
import gym_smartmeter
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pandas as pd
import math

df_loaded = pd.read_csv('D:\\Project\\smartmeter0\\UKData_1day_clean\\sum_all_1d.csv', index_col=0)

battery_capacity = 90000
total_steps = 1440
origin_values = df_loaded.loc['Total Usage'].tolist()



# Hyperparameters
BUFFER_SIZE_I = 10000  # Size of the experience replay memory I
BUFFER_SIZE_II = 500  # Size of the experience replay memory II
BATCH_SIZE_Q = 128  # Batch size for Q-network update
BATCH_SIZE_H = 64  # Batch size for H-network update
COPY_STEPS = 500  # Target-network copy steps
TRAINING_STEPS = 8  # Training steps for Q-network update
LEARNING_RATE_Q = 0.00025
LEARNING_RATE_H = 0.001
T_MAX = 1440 # Max steps in an episode
NUM_EPISODES = 500  # Number of training episodes


# Define Q-network and Target-network architecture
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.output_layer(x)


# Define H-network architecture
class HNetwork(nn.Module):
    def __init__(self, input_size, sequence_length):
        super(HNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64 * 2, 1)  # Bidirectional, so output is doubled
        self.tanh = nn.Tanh()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Take the output of the last sequence element
        return self.tanh(x)


# Create the environment

env = gym.make('smartmeter-v0',
               total_steps=total_steps,
               battery_capacity=battery_capacity,
               origin_values=origin_values)

n_actions = env.action_space.n
state_dim = env.observation_space.shape[0]

# Initialize networks
q_network = MultiLayerPerceptron(state_dim, n_actions)
target_network = MultiLayerPerceptron(state_dim, n_actions)
h_network = HNetwork(input_size=state_dim, sequence_length=T_MAX)

# Initialize replay buffers
replay_buffer_I = deque(maxlen=BUFFER_SIZE_I)
replay_buffer_II = deque(maxlen=BUFFER_SIZE_II)

# Optimizers
optimizer_q = optim.RMSprop(q_network.parameters(), lr=LEARNING_RATE_Q)
optimizer_h = optim.RMSprop(h_network.parameters(), lr=LEARNING_RATE_H)

# Loss function for Q-network
loss_fn = nn.MSELoss()


# Function to update the network
def train_q_network(q_network, optimizer_q, batch_size, replay_buffer_I):
    if len(replay_buffer_I) < batch_size:
        return
    batch = random.sample(replay_buffer_I, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to PyTorch tensors
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Compute the target Q values
    current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    max_next_q_values = target_network(next_states).detach().max(1)[0]
    expected_q_values = rewards + 0.99 * max_next_q_values * (1 - dones)  # discount factor gamma=0.99

    # Compute loss
    loss = loss_fn(current_q_values, expected_q_values)

    # Step optimizer
    optimizer_q.zero_grad()
    loss.backward()
    optimizer_q.step()


def train_h_network(h_network, optimizer_h, batch_size, replay_buffer_II):
    if len(replay_buffer_II) < batch_size:
        return 0

    minibatch = random.sample(replay_buffer_II, batch_size)
    states_sequences, z_sequences = zip(*minibatch)

    # 将状态序列和 z 序列处理成合适的格式
    states_sequences_tensor = torch.stack([torch.tensor(s) for s in states_sequences]).float()
    z_sequences_tensor = torch.stack([torch.tensor(z) for z in z_sequences]).float()

    # H-network 输入为 states_sequences_tensor 和 z_sequences_tensor
    # 假设 states_sequences_tensor 包含 Y^{t-1} 和 S_t，z_sequences_tensor 包含 Z^T
    inputs = torch.cat((states_sequences_tensor, z_sequences_tensor), 2)

    # 目标为 0，即我们试图让网络输出接近 0，这最小化了隐私泄露信号
    targets = torch.zeros(batch_size, 1)

    # 前向传播
    optimizer_h.zero_grad()
    outputs = h_network(inputs)

    # 计算损失
    loss = nn.MSELoss()(outputs, targets)

    # 反向传播和优化
    loss.backward()
    optimizer_h.step()

    return loss.item()


# Assuming that 'replay_buffer_II' contains tuples (Z^T, Y^T) for the full sequence T=1440
# Here, we convert these to tensors and concatenate them into the right shape
# The following code is conceptual and assumes each Z_t and Y_t are already vectors of the same length
# and 'input_size' is set to the length of these vectors

def prepare_sequences_for_h_network(batch):
    # This function would reshape and prepare the input batch for the H-network
    sequences = []
    for z_sequence, y_sequence in batch:
        # Here, we concatenate the Z_t and Y_t vectors for each time step t in the sequence
        # We then stack these to create a sequence tensor for each sample in the batch
        sequence_tensor = torch.cat([
            torch.tensor(z_t).unsqueeze(0) + torch.tensor(y_t).unsqueeze(0)
            for z_t, y_t in zip(z_sequence, y_sequence)
        ], dim=0)
        sequences.append(sequence_tensor)

    # Stack all sequence tensors to create a batch
    # The shape of 'input_batch' should be (batch_size, sequence_length, input_size)
    input_batch = torch.stack(sequences, dim=0)
    return input_batch


def calculate_reward(h_network, replay_buffer_II, current_state, current_action, current_time_step):
    """
    计算给定状态和动作的奖励值。

    Args:
        h_network: 训练好的 H-network。
        replay_buffer_II: 包含 (Y^T, Z^T) 样本的缓冲区 II。
        current_state: 当前状态，这里是 y_t。
        current_action: 当前动作，这里是 z_t - y_t。
        current_time_step: 当前的时间步，用于从 Y^T 中索引 Y_t。

    Returns:
        float: 估计的隐私泄露信号作为奖励值。
    """
    # 从 BUFFER II 中选择一个历史序列
    # 这里我们选择最新的历史序列以保持信息的实时性
    y_sequence, z_sequence = replay_buffer_II[-1]  # 假设 replay_buffer_II 以正确的格式存储数据

    # 为 H-network 准备输入
    # 注意这里的 y_sequence 应该包含从 0 到 t-1 的值，而 z_sequence 包含整个序列
    y_sequence_t_minus_1 = torch.tensor(y_sequence[:current_time_step], dtype=torch.float32)
    z_sequence_full = torch.tensor(z_sequence, dtype=torch.float32)

    # 注意这里的 current_state 和 current_action 必须是 float tensor
    current_state_tensor = torch.tensor([current_state], dtype=torch.float32)
    current_action_tensor = torch.tensor([current_action], dtype=torch.float32)

    # 将它们组合成适当的输入形式
    h_input = torch.cat((y_sequence_t_minus_1, z_sequence_full, current_state_tensor, current_action_tensor))
    h_input = h_input.unsqueeze(0)  # 添加一个批次维度

    # 使用 H-network 估计隐私泄露信号
    h_network.eval()  # 设置为评估模式
    with torch.no_grad():
        privacy_leakage_signal = h_network(h_input)

    # 将信号转换为奖励值，取反因为我们想要最小化隐私泄露
    reward = -privacy_leakage_signal.item()
    return reward


# Main training loop
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    episode_reward = 0


    for t in range(T_MAX):
        # Select an action
        epsilon = max(0.01, 0.08 - 0.01 * (episode / 200))  # Linearly decreasing epsilon
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(torch.tensor(state, dtype=torch.float32))
                action = q_values.argmax().item()


        # Take the action and observe the reward and next state
        next_state, _, done, _, _ = env.step(action)
        # obs, reward, done, truncated, info = env.step(action)
        reward = calculate_reward(h_network,
                                  replay_buffer_II,
                                  current_state,
                                  current_action,
                                  current_time_step)

        # Store in replay buffer I
        replay_buffer_I.append((state, action, reward, next_state, done))

        # Update Q-network
        if len(replay_buffer_I) > BATCH_SIZE_Q and t % TRAINING_STEPS == 0:
            train_q_network(q_network, optimizer_q, BATCH_SIZE_Q, replay_buffer_I)

        # Update target network
        if t % COPY_STEPS == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Update H-network (training logic should be implemented here based on the algorithm)
        if len(replay_buffer_II) > BATCH_SIZE_H and t % COPY_STEPS == 0:
            train_h_network(h_network, optimizer_h, BATCH_SIZE_H, replay_buffer_II)

        # Move to the next state
        state = next_state
        if done:
            break

    replay_buffer_II.append((y_T, z_T))
    print(f"Episode {episode}: Total Reward: {episode_reward}")

env.close()
