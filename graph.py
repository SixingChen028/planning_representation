import numpy as np
import matplotlib.pyplot as plt


class Graph:
    """
    A graph class.
    """

    def __init__(self, adj_list = None, adj_matrix = None, rewards = None):
        """
        Initialize the graph.
        """
        
        if adj_list is None and adj_matrix is None:
            raise ValueError('Graph is not provided.')
        
        # initialize with adjacency matrix
        elif adj_matrix is not None and adj_list is None:
            self.adj_matrix = adj_matrix
            self.adj_list = []
            for i in range(self.adj_matrix.shape[0]):
                connected_states = []
                for j in range(self.adj_matrix.shape[1]):
                    if self.adj_matrix[i][j] == 1:
                        connected_states.append(j)
                self.adj_list.append(connected_states)
        
        # initialize with adjacency list
        elif adj_list is not None and adj_matrix is None:
            self.adj_list = adj_list
            self.adj_matrix = np.zeros((len(adj_list), len(adj_list)))
            for i in range(len(adj_list)):
                for j in self.adj_list[i]:
                    self.adj_matrix[i, j] = 1
        
        # initialize rewards if provided
        if rewards is not None:
            self.reset_rewards(rewards)


    def reset_rewards(self, rewards):
        """
        Reset rewards.
        """
        self.rewards = rewards
        
    
    def successors(self, state):
        """
        Find successor states of a given state.
        """
        return self.adj_list[state]


    def predecessors(self, state):
        """
        Find predecessor states of a given state.
        """
        return [s for s, successors in enumerate(self.adj_list) if state in successors]
    

    def max_reward_path(self, state, steps_left = 5, cum_reward = np.array(0), path = None):
        """
        Find the maximum possible sum of reward within 5 steps.
        """

        if path is None:
            path = []

        path.append(state)
        if steps_left < 5:
            cum_reward += self.rewards[state]

        if steps_left == 0:
            return cum_reward, path
        
        cum_reward_successor1, path_successor1 = self.max_reward_path(self.successors(state)[0], steps_left - 1, cum_reward.copy(), path.copy())
        cum_reward_successor2, path_successor2 = self.max_reward_path(self.successors(state)[1], steps_left - 1, cum_reward.copy(), path.copy())

        if cum_reward_successor1 >= cum_reward_successor2:
            max_cum_reward = cum_reward_successor1
            max_path = path_successor1
        else:
            max_cum_reward = cum_reward_successor2
            max_path = path_successor2


        return max_cum_reward.copy(), max_path.copy()
        





if __name__ == '__main__':
    # testing

    num_state = 32

    adj_matrix = np.zeros((num_state, num_state))
    for i in range(num_state):
        row, col1, col2 = i, (i + 1) % num_state, (i + 2) % num_state

        adj_matrix[row, col1] = 1
        adj_matrix[row, col2] = 1

    adj_list = []
    for i in range(adj_matrix.shape[0]):
        connected_states = []
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i][j] == 1:
                connected_states.append(j)
        adj_list.append(connected_states)

    graph = Graph(adj_matrix = adj_matrix)
    
    print(graph.adj_matrix)
    print(graph.adj_list)
    print(graph.predecessors(0))

    plt.figure()
    plt.imshow(graph.adj_matrix)
    plt.show()
