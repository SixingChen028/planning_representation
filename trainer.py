import numpy as np
import time

import gymnasium as gym

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


class A2C:
    """
    An A2C trainer.
    """

    def __init__(
            self,
            net,
            env,
            lr,
            gamma,
            beta_v,
            beta_e,
            beta_p,
            lr_schedule = None,
            entropy_schedule = None,
            max_grad_norm = None
            
        ):
        """
        Initialize the trainer.
        """

        self.net = net
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.beta_v = beta_v
        self.beta_e = beta_e
        self.beta_p = beta_p
        self.lr_schedule = lr_schedule
        self.entropy_schedule = entropy_schedule
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(net.parameters(), lr = self.lr)


    def get_discounted_returns(self, rewards, values):
        """
        Compute discounted reterns and advantages.
        """

        # initialize recordings
        returns = []
        advantages = []

        # compute returns and advantages from the last timestep
        R = values[-1].item()
        advantage = 0
        
        for i in reversed(range(len(rewards))):
            # get (v, r, v')
            r = rewards[i]
            v = values[i]
            v_next = values[i + 1]

            # compute return for the timestep
            R = r + R * self.gamma

            # compute advantage for the timestep
            delta = r + v_next * self.gamma - v
            advantage = advantage * self.gamma + delta

            # insert 0 for the previous timestep
            returns.insert(0, R)
            advantages.insert(0, advantage)
            
        return torch.Tensor(returns), torch.Tensor(advantages)
    

    def update_model(
            self,
            rewards,
            values,
            log_probs,
            entropies,
            states,
            state_predictions,
            reward_predictions,
        ):
        """
        Update model parameters.
        """

        # compute returns and advantages
        returns, advantages = self.get_discounted_returns(rewards, values)

        # compute policy loss
        policy_loss = -(log_probs * advantages.detach()).sum()

        # compute value loss
        value_loss = F.mse_loss(values[:-1], returns, reduction = 'sum')

        # compute entropy loss
        entropy_loss = -entropies.sum()

        # compute prediction loss
        state_prediction_loss = F.cross_entropy(state_predictions, states, reduction = 'sum')
        reward_prediction_loss = F.mse_loss(reward_predictions, rewards, reduction = 'sum')

        # compute loss
        loss = policy_loss + self.beta_v * value_loss + self.beta_e * entropy_loss + self.beta_p * (state_prediction_loss + reward_prediction_loss)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item(), state_prediction_loss.item(), reward_prediction_loss.item()
    

    def train_one_episode(self):
        """
        Train one episode.
        """

        # initialize recordins
        actions = []
        values = []
        log_probs = []
        entropies = []
        rewards = []
        reward_predictions = []
        states = []
        state_predictions = []
        
        # initialize a trial
        done = False
        states_actor, states_critic = None, None

        obs, info = self.env.reset()
        obs = torch.Tensor(obs).unsqueeze(dim = 0)

        # iterate through a trial
        while not done:
            # step the net
            action, policy, log_prob, entropy, value, state_prediction, reward_prediction, states_actor, states_critic = self.net(obs, states_actor, states_critic, mask = info['mask'])

            # step the env
            obs, reward, done, truncated, info = self.env.step(action.item())
            obs = torch.Tensor(obs).unsqueeze(dim = 0)

            # record results
            actions.append(action.item())
            values.append(value.view(-1))
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            reward_predictions.append(reward_prediction.view(-1))
            states.append(info['state'])
            state_predictions.append(state_prediction)

        # process the last timestep
        action, policy, log_prob, entropy, value, state_prediction, reward_prediction, states_actor, states_critic = self.net(obs, states_actor, states_critic, mask = info['mask'])
        values.append(value.view(-1)) # values have one more step

        # concatenate recordings
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        entropies = torch.stack(entropies)
        
        rewards = torch.Tensor(rewards)
        reward_predictions = torch.cat(reward_predictions)

        states = torch.LongTensor(states)
        state_predictions = torch.cat(state_predictions)

        # update model
        loss, policy_loss, value_loss, entropy_loss, state_prediction_loss, reward_prediction_loss = self.update_model(rewards, values, log_probs, entropies, states, state_predictions, reward_predictions)

        # compute reward and length of an epiosde
        episode_reward = rewards.sum()
        episode_length = len(rewards)

        return loss, policy_loss, value_loss, entropy_loss, state_prediction_loss, reward_prediction_loss, episode_reward, episode_length


    def learn(self, num_episodes, print_frequency = 2):
        """
        Train the model.
        """

        # initialize recordings
        data = {
            'episode_length': [],
            'episode_reward': [],
            'loss': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'state_pred_loss': [],
            'reward_pred_loss': [],
        }

        # train the model
        start_time = time.time()
        for episode in range(num_episodes):
            # train the model for one episode
            loss, policy_loss, value_loss, entropy_loss, state_prediction_loss, reward_prediction_loss, episode_reward, episode_length = self.train_one_episode()
        
            # record training data
            data_episode = {
                'episode_length': episode_length,
                'episode_reward': episode_reward,
                'loss': loss,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'entropy_loss': entropy_loss,
                'state_pred_loss': state_prediction_loss,
                'reward_pred_loss': reward_prediction_loss,
            }
            for key, item in data_episode.items():
                data[key].append(data_episode[key])

            # print the training process
            self.print_training_process(
                ep_num = episode + 1,
                time_elapsed = time.time() - start_time,
                data = data_episode,
                print_frequency = print_frequency
            )

            # update learning rate
            if self.lr_schedule is not None:
                self.update_learning_rate(episode)

            # update entropy regularization
            if self.entropy_schedule is not None:
                self.update_entropy_coef(episode)
        
        return data
    

    def update_learning_rate(self, episode):
        """
        Update the learning rate based on the episode number.
        """

        if episode < len(self.lr_schedule):
            self.lr = self.lr_schedule[episode]

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
    

    def update_entropy_coef(self, episode):
        """
        Update the entropy regularization coefficient based on the episode number.
        """

        if episode < len(self.entropy_schedule):
            self.beta_e = self.entropy_schedule[episode]


    def save_net(self, path):
        """
        Save the net.
        """
        
        torch.save(self.net, path)


    def print_training_process(self, ep_num, time_elapsed, data, print_frequency = 2):
        """
        Print the training process.
        """

        if ep_num % print_frequency == 0:
            print("-------------------------------------------")
            print("| rollout/                |               |")
            print(f"|    ep_len_mean          | {data['episode_length']:<13} |")
            print(f"|    ep_rew_mean          | {data['episode_reward']:<13} |")
            print("| time/                   |               |")
            print(f"|    ep_num               | {ep_num:<13} |")
            print(f"|    time_elapsed         | {time_elapsed:<13.4f} |")
            print("| train/                  |               |")
            print(f"|    learning_rate        | {self.lr:<13.5f} |")
            print(f"|    loss                 | {data['loss']:<13.4f} |")
            print(f"|    policy_loss          | {data['policy_loss']:<13.4f} |")
            print(f"|    value_loss           | {data['value_loss']:<13.4f} |")
            print(f"|    entropy_loss         | {data['entropy_loss']:<13.4f} |")
            print(f"|    state_pred_loss      | {data['state_pred_loss']:<13.4f} |")
            print(f"|    reward_pred_loss     | {data['reward_pred_loss']:<13.4f} |")
            print("-------------------------------------------")



