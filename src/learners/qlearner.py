import random

import numpy as np

from src.environments.models import ACTIONS


class QLearner:
    def __init__(self, grid_shape, n_actions=5, learning_rate=0.3, eps=0.01, discount=.9):
        self.Q = {(row_Theseus, col_Theseus, row_Minautor, col_Minautor): [0 for _ in range(n_actions)]
                  for row_Theseus in range(-1, grid_shape[0] + 1)
                  for col_Theseus in range(-1, grid_shape[1] + 1)
                  for row_Minautor in range(-1, grid_shape[0] + 1)
                  for col_Minautor in range(-1, grid_shape[1] + 1)}
        self.learning_rate = learning_rate
        self.eps = eps
        self.discount = discount

    def choose_action_train(self, state, observation=None):
        q = self.Q[state]
        if random.random() < self.eps:
            action = random.choice(ACTIONS)

        else:
            action = np.argmax(q)

        return action

    def choose_action_best(self, state):
        return ACTIONS[np.argmax(self.Q[state])]

    def learn(self, state, action, reward, next_state):
        self.Q[state][action] = ((1 - self.learning_rate) * self.Q[state][action] +
                                 self.learning_rate * (reward + self.discount * np.max(self.Q[next_state])))
