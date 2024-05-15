import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch

import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from graph import *

# from sb3_contrib import RecurrentPPO

class GraphEnv(gym.Env):
    """
    A graph environment.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
            self,
            num_state,
            reward_set,
            adj_list = None,
            adj_matrix = None,
            t_ponder = 5,
            t_act = 5,
        ):
        """
        Construct an environment.
        """

        self.num_state = num_state
        self.reward_set = reward_set

        self.t_ponder = t_ponder
        self.t_act = t_act

        # initialize the graph
        self.graph = Graph(adj_list = adj_list, adj_matrix = adj_matrix)

        # initialize action space
        self.action_space = Discrete(3)

        # initialize observation space
        observation_shape = (
            self.num_state + # current state (num_state,)
            self.num_state + # rewards at all states (num_state,)
            2, # timer and stage
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = observation_shape,)


    def reset(self, seed = None, option = {}):
        """
        Reset the environment.
        """

        # reset timer and stage
        self.timer = 0
        self.stage = 0
        
        # permtute reward set to get rewards
        rewards = np.random.permutation(self.reward_set)
        self.graph.reset_rewards(rewards)

        # randomly initialize the starting state
        self.state = np.random.randint(0, self.num_state)

        # wrap observation
        obs = np.hstack([
            self.one_hot_coding(num_classes = self.num_state, labels = self.state),
            self.graph.rewards,
            self.timer,
            self.stage,
        ])

        # get info
        info = {
            'state': self.state,
            'rewards': self.graph.rewards,
            'stage': self.stage,
            'mask': self.get_action_mask(),
        }

        return obs, info
    

    def step(self, action):
        """
        Step the environment.
        """

        self.timer += 1
        done = False
        reward = 0.

        # pondering stage
        if self.stage == 0:
            if action != 2:
                raise ValueError('Execute decision action in pondering stage.')

        # decision stage
        elif self.stage == 1:
            if action != 2:
                # state transition: (s, a) -> (s', r)
                self.state = self.graph.successors(self.state)[action] # ignore repetitive visiting
                reward = self.graph.rewards[self.state]
        
            elif action == 2:
                raise ValueError('Execute pondering action in decision stage.')

        # stage transition
        if self.timer == self.t_ponder:
            self.stage = 1
        
        # end a trial
        if self.timer == self.t_ponder + self.t_act:
            done = True

        # wrap observation
        obs = np.hstack([
            self.one_hot_coding(num_classes = self.num_state, labels = self.state),
            self.graph.rewards,
            self.timer,
            self.stage,
        ])

        # get info
        info = {
            'state': self.state,
            'rewards': self.graph.rewards,
            'stage': self.stage,
            'mask': self.get_action_mask(),
        }
        
        return obs, reward, done, False, info
    

    def get_action_mask(self):
        """
        Get action mask.
        """
        batch_size = 1
        mask = torch.zeros((batch_size, self.action_space.n), dtype = torch.bool)

        # pondering stage
        if self.stage == 0:
            mask[0, 2] = True

        # decision stage
        elif self.stage == 1:
            mask[0, 0] = True
            mask[0, 1] = True
        
        return mask


    def one_hot_coding(self, num_classes, labels = None):
        """
        One-hot code states.
        """

        if labels is None:
            labels_one_hot = np.zeros((num_classes,))
        else:
            labels_one_hot = np.eye(num_classes)[labels]

        return labels_one_hot


    
class MetaLearningWrapper(Wrapper):
    """
    A meta-RL wrapper.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, env):
        """
        Construct an wrapper.
        """

        super().__init__(env)

        self.env = env
        self.num_state = env.get_wrapper_attr('num_state')
        self.one_hot_coding = env.get_wrapper_attr('one_hot_coding')

        # initialize previous variables
        self.init_prev_variables()

        # define new observation space
        new_observation_shape = (
            self.env.observation_space.shape[0] + # obs
            self.env.action_space.n + # previous action
            1, # previous reward
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = new_observation_shape)


    def step(self, action):
        """
        Step the environment.
        """

        obs, reward, done, truncated, info = self.env.step(action)

        # concatenate previous variables into observation
        obs_wrapped = self.wrap_obs(obs)

        # update previous variables
        self.prev_action = action
        self.prev_reward = reward

        return obs_wrapped, reward, done, truncated, info
    

    def reset(self, seed = None, options = {}):
        """
        Reset the environment.
        """

        obs, info = self.env.reset()

        # initialize previous physical action and reward
        self.init_prev_variables()

        # concatenate previous physical action and reward into observation
        obs_wrapped = self.wrap_obs(obs)

        return obs_wrapped, info
    

    def init_prev_variables(self):
        """
        Reset previous variables.
        """

        self.prev_action = None
        self.prev_reward = 0.


    def wrap_obs(self, obs):
        """
        Wrap observation with previous variables.
        """

        obs_wrapped = np.hstack([
            obs, # current obs
            self.one_hot_coding(num_classes = self.env.action_space.n, labels = self.prev_action),
            self.prev_reward,
        ])
        return obs_wrapped
    




if __name__ == '__main__':

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


    env = GraphEnv(
        num_state = 32,
        reward_set = np.concatenate([np.repeat(1, 10), np.repeat(-1, 21), np.array([5])]),
        adj_matrix = adj_matrix,
    )
    env = MetaLearningWrapper(env)
    
    for i in range(50):

        obs, info = env.reset()
        done = False

        print('initial obs:', obs.shape)
        print('init state:', env.state)
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            print(
                'obs:', obs.shape, '|',
                'action:', action, '|',
                'reward:', reward, '|',
                'state:', env.state, '|',
                'timer:', env.timer, '|',
                'done:', done, '|',
            )
        print()
    


