import numpy as np
import time
import pickle
import argparse

import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from graph import *
from environment import *

from environment import *
from modules import *
from trainer import *

# parse args
parser = argparse.ArgumentParser()

parser.add_argument('--num_episodes', type = int, default = 3500000, help = 'training episodes')

parser.add_argument('--hidden_size', type = int, default = 128, help = 'lstm hidden size')
parser.add_argument('--policy_hidden_size', type = int, default = 32, help = 'policy head hidden size')
parser.add_argument('--value_hidden_size', type = int, default = 32, help = 'value head hidden size')
parser.add_argument('--prediction_hidden_size', type = int, default = 32, help = 'prediction head hidden size')

parser.add_argument('--t_ponder', type = int, default = 5, help = 'pondering time steps')
parser.add_argument('--t_act', type = int, default = 5, help = 'acting time steps')

parser.add_argument('--lr', type = float, default = 3e-4, help = 'learning rate')
parser.add_argument('--gamma', type = float, default = 0.9, help = 'temporal discount')
parser.add_argument('--beta_v', type = float, default = 0.05, help = 'value loss coefficient')
parser.add_argument('--beta_e', type = float, default = 0.05, help = 'entropy regularization coefficient')
parser.add_argument('--beta_p', type = float, default = 0.5, help = 'predictive loss coefficient')

args = parser.parse_args()

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
    t_ponder = args.t_ponder,
    t_act = args.t_act,
)
env = MetaLearningWrapper(env)

# set net
net = RecurrentActorCriticPolicy(
    feature_dim = env.observation_space.shape[0],
    action_dim = env.action_space.n,
    state_dim = env.num_state,
    lstm_hidden_dim = args.hidden_size,
    policy_hidden_dim = args.policy_hidden_size,
    value_hidden_dim = args.value_hidden_size,
    prediction_hidden_dim = args.prediction_hidden_size,
)

# network training
a2c = A2C(
    net = net,
    env = env,
    lr = args.lr,
    gamma = args.gamma,
    beta_v = args.beta_v,
    beta_e = args.beta_e,
    beta_p = args.beta_p,
    # lr_schedule = np.linspace(3e-4, 1e-4, num = 30000),
)

# save data
data = a2c.learn(num_episodes = args.num_episodes, print_frequency = 100)
a2c.save_net(path + '/net.pth')
pickle.dump(data, open(path + '/data.p', 'wb'))
