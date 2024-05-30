import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

from copy import deepcopy
import pandas as pd
import numpy as np

def index_to_action(action_index):
    action = 20 * (action_index - 250)
    return action

def action_to_index(action):
    action_index = action//20 + 250
    return action_index

class SmartMeterBoxEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, total_steps, battery_capacity, origin_values, trade_off_factor):
        super(SmartMeterBoxEnv,self).__init__()

        # Define observation space
        self.observation_space = spaces.Box(low=np.array([0, 0, 0], dtype=np.float32),
                                            high=np.array([5999, 90000, 1439], dtype=np.float32),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(501) # Battery rate: 5kW

        self.total_steps = total_steps
        self.current_step = 0
        self.battery_capacity = battery_capacity
        self.state_of_charge = [0 for i in range(self.total_steps)]
        self.origin_values = origin_values # Predicted load demand
        self.sliding_curve = deepcopy(origin_values)
        self.sliding_avg = [0 for i in range(self.total_steps)]
        self.sliding_avg[0] = self.sliding_curve[0]
        self.trade_off_factor = trade_off_factor

    def normalize_observation(self, obs):
        obs_min = np.array([0, 0, 0], dtype=np.float32)
        obs_max = np.array([5999, 90000, 1439], dtype=np.float32)
        normalized_obs = (obs - obs_min) / (obs_max - obs_min)
        return normalized_obs.astype(np.float32)

    def is_illegal_action(self, action_index):
        action = index_to_action(action_index)
        load_level = self.origin_values[self.current_step]
        battery_remain = self.state_of_charge[self.current_step]
        action_high = min(load_level, battery_remain)
        action_low = battery_remain - self.battery_capacity
        #
        if action > action_high or action < action_low:
            return True
        else:
            return False

    def action_mask(self, action_index):
        action = index_to_action(action_index)
        load_level = self.origin_values[self.current_step]
        battery_remain = self.state_of_charge[self.current_step]
        action_low = battery_remain - self.battery_capacity
        action_high = min(load_level, battery_remain)
        if action >= 0:  # battery discharging
            '''
            action >= 0, battrey is discharging, load level is smaller
            Constraint:
            1. discharging amount can not exceed battery capacity
            2. discharging amount can not exceed load level
            '''
            action = min(action, action_high)

        elif action < 0:  # battery charging
            '''
            action < 0, battery is charging, load level is higher
            Constraint:
            1. charging amount can not exceed rest of the battery capacity
            '''
            action = max(action, action_low)

        return action

    def cost_saving(self, action):
        if 0 <= self.current_step%1440 <= 419:
            price_kwh = 13.22
        if 420 <= self.current_step%1440 <=1440:
            price_kwh = 30.41
        price_wmin = price_kwh/60000
        price_wmin_high = 30.41/60000
        price_wmin_min = 13.22/60000
        # action > 0, battery discharging
        return action * price_wmin

    def normalized_cost_saving(self, action):
        if 0 <= self.current_step%1440 <= 419:
            price_kwh = 13.22
        if 420 <= self.current_step%1440 <= 1440:
            price_kwh = 30.41
        price_wmin = price_kwh/60000
        price_wmin_high = 30.41/60000
        price_wmin_min = 13.22/60000
        # action > 0, battery discharging
        return (action * price_wmin + 5000 * price_wmin_high)/(2 * 5000 * price_wmin_high)

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)

        self.masked_values = deepcopy(self.origin_values)
        self.current_step = 0
        self.state_of_charge = [0 for i in range(self.total_steps)]
        self.state_of_charge[0] = self.battery_capacity
        self.sliding_avg = [0 for i in range(self.total_steps)]
        self.sliding_avg[0] = self.sliding_curve[0]
        self.count_masked = 0
        self.count_unmasked = 0
        self.count_artificial = 0
        self.peak = [0 for i in range(self.total_steps)]
        self.initial_state = np.array([int(self.origin_values[0]),
                        self.state_of_charge[0],
                        self.current_step], dtype=np.float32)
        normalized_state = self.normalize_observation(self.initial_state)
        info={}

        return normalized_state, info

    def step(self, action_index):
        # action > 0: battery discharging
        # action < 0: battery charging
        truncated = False
        # Illegal action
        if self.is_illegal_action(action_index):
            action_from_agent = index_to_action(action_index)
            action = self.action_mask(action_index)
            info = {'Valid action': False, 'Legal action': action}
            reward_from_illegal_action = - abs((action-action_from_agent)/action_from_agent)
        else:
            reward_from_illegal_action = 0
            action = index_to_action(action_index)
            info = {'Valid action': True, 'Legal action': action}

        normalized_cost_saving = self.normalized_cost_saving(action)

        # Sliding window
        sliding_length = min(10, self.current_step)
        sliding_data = self.sliding_curve[self.current_step-sliding_length:self.current_step]
        if sliding_length == 0:
            sliding_data = [self.sliding_curve[0]]
        self.sliding_avg[self.current_step] = np.mean(sliding_data)
        sliding_std = np.std(sliding_data)
        threshold = self.sliding_avg[self.current_step] + 3*sliding_std

        # Last step
        if self.current_step >= self.total_steps-1:
            self.masked_values[self.current_step] = self.origin_values[self.current_step] - action
            # self.state_of_charge[self.current_step+1] = self.state_of_charge[self.current_step] - action
            state_of_charge_last = self.state_of_charge[self.current_step] - action
            done = True
            battery_penalty = -abs(self.state_of_charge[self.current_step]-self.battery_capacity)
            if battery_penalty==0:
                normalized_battery_penalty = 0
            else:
                normalized_battery_penalty = battery_penalty / self.battery_capacity
            reward = 100 * normalized_battery_penalty
            # MI = normalized_mutual_info_score(self.masked_values, self.origin_values)
            # reward = -80*MI+0.1*battery_penalty
            info['Masked'] = self.masked_values[self.current_step]
            print(f'Masked peaks:{self.count_masked},Not masked peaks:{self.count_unmasked},Artificial peaks:{self.count_artificial}')
            # print(f'MI:{MI}')
        # Not last step
        else:
            done = False
            self.masked_values[self.current_step] = self.origin_values[self.current_step] - action
            self.state_of_charge[self.current_step+1] = self.state_of_charge[self.current_step] - action
            info['Masked'] = self.masked_values[self.current_step]

            # If there is a peak
            if self.origin_values[self.current_step] >= max(threshold, 500):
                self.sliding_curve[self.current_step] = self.sliding_avg[self.current_step]
                self.peak[self.current_step] = self.origin_values[self.current_step]

                # Hiding peaks
                if self.masked_values[self.current_step] < self.sliding_avg[self.current_step]:
                    reward_privacy = 100 + \
                                     0.05 * (self.origin_values[self.current_step] - threshold)
                    self.count_masked += 1
                # Not hiding peaks
                else:
                    reward_privacy = -50 - 0.05 * \
                                     (self.masked_values[self.current_step] - self.sliding_avg[self.current_step])
                    self.count_unmasked += 1

            # If there is no peak
            elif self.origin_values[self.current_step] < max(threshold, 500):
                # Successful artificial peaks
                if self.masked_values[self.current_step] > max(threshold,800):
                    reward_privacy = 20 + \
                                     0.05*(self.masked_values[self.current_step] - threshold)
                    self.count_artificial += 1
                # Not artificial peaks
                else:
                    reward_privacy = -20
            else:
                reward_privacy = 0


            normalized_reward_privacy = (reward_privacy + 250) / 550
            reward = self.trade_off_factor * normalized_reward_privacy + \
                     (1-self.trade_off_factor) * normalized_cost_saving + \
                     0.5 * reward_from_illegal_action
            done = False
            self.current_step += 1

        state = np.array([
            int(self.origin_values[self.current_step]),
            int(self.state_of_charge[self.current_step]),
            int(self.current_step)
        ])
        normalized_state = self.normalize_observation(state)
        # info['Masked'] = np.array(self.masked_values[self.current_step])

        # print(info['Masked'],type(info['Masked']))
        return normalized_state, reward, done, truncated,info

    def render(self, mode='human'):
        if mode == 'human':
            print("Current State:", {
            "time": self.current_step,  # time
            "load level": self.origin_values[self.current_step],  # Load level
            "SoC": self.state_of_charge[self.current_step],  # State of charge
        })

# df_loaded = pd.read_csv('D:\\Project\\smartmeter0\\UKData_1day_clean\\sum_all_1d.csv', index_col=0)
# battery_capacity = 5000
# total_steps = 1440
# origin_values = df_loaded.loc['Total Usage'].tolist()
# origin_curve = df_loaded.loc['Total Usage'].tolist()
# LAMBDA = 1
# env = SmartMeterBoxEnv(total_steps=total_steps, battery_capacity=battery_capacity, origin_values=origin_values, trade_off_factor=LAMBDA)
# print(env.reset())