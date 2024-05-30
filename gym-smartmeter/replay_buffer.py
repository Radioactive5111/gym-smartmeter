import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def store_transition(self, masked_value, origin_value):
        transition = (masked_value, origin_value)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        masked_value, origin_value = zip(*batch)
        return masked_value, origin_value
    
    def __len__(self):
        return len(self.buffer)


# def test_replay_buffer():
#     replay_buffer = ReplayBuffer(buffer_size=1000)
#
#     # 存储一些转换样本
#     for i in range(10):
#         state = i
#         action = i + 1
#         reward = i + 2
#         next_state = i + 3
#         done = False
#         replay_buffer.store_transition(state, action, reward, next_state, done)
#
#     # 从回放缓冲区中采样一批转换样本
#     batch_size = 4
#     states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
#
#     # 打印采样结果
#     print("Sampled Batch:")
#     for i in range(batch_size):
#         print(f"State: {states[i]}, Action: {actions[i]}, Reward: {rewards[i]}, Next State: {next_states[i]}, Done: {dones[i]}")
#
#     # 打印回放缓冲区的大小
#     print(f"Replay Buffer Size: {len(replay_buffer)}")
#
# if __name__ == "__main__":
#     test_replay_buffer()
