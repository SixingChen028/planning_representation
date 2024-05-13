import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from graph import *
from environment import *

from environment import *
from modules import *
from trainer import *


# parameters
num_episodes = 1000000
path = '/home/sc10264/samplingrnn/code_planning_representation/results'

# create adjacency matrix
adj_dict = {
    0: [9, 27], 1: [22, 28], 2: [22, 10], 3: [2, 23],
    4: [14, 19], 5: [12, 21], 6: [2, 8], 7: [15, 31],
    8: [16, 20], 9: [11, 25], 10: [0, 30], 11: [5, 25],
    12: [0, 7], 13: [26, 28], 14: [13, 29], 15: [6, 29],
    16: [17, 19], 17: [21, 24], 18: [30, 31], 19: [13, 23],
    20: [3, 4], 21: [3, 8], 22: [7, 9], 23: [1, 18],
    24: [4, 11], 25: [6, 15], 26: [12, 18], 27: [5, 20],
    28: [1, 14], 29: [24, 26], 30: [17, 27], 31: [10, 16],
}
adj_matrix = np.zeros((32, 32))
for key, item in adj_dict.items():
    adj_matrix[key, item[0]] = 1
    adj_matrix[key, item[1]] = 1

# set environment
env = GraphEnv(
    num_state = 32,
    reward_set = np.concatenate([np.repeat(1, 10), np.repeat(-1, 21), np.array([5])]),
    adj_matrix = adj_matrix,
)
env = MetaLearningWrapper(env)

# set network
net = RecurrentActorCriticPolicy(
    feature_dim = env.observation_space.shape[0],
    action_dim = env.action_space.n,
    state_dim = env.num_state,
    lstm_hidden_dim = 128,
    policy_hidden_dim = 32,
    value_hidden_dim = 32,
    prediction_hidden_dim = 32,
)

# training
a2c = A2C(
    net = net,
    env = env,
    lr = 3e-4,
    gamma = 0.9,
    beta_v = 0.05,
    beta_e = 0.05,
    beta_p = 0.5,
    # lr_schedule = np.linspace(3e-4, 1e-4, num = 30000),
)

# save training results
data = a2c.learn(num_episodes = 2000000)
a2c.save_net(path + '/net.pth')
pickle.dump(data, open(path + '/data.p', 'wb'))
