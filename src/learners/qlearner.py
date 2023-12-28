import random

import numpy as np

from src.environments.models import ACTIONS


class QLearner:
    def __init__(
        self, grid_shape, n_actions=5, learning_rate=0.3, eps=0.01, discount=0.9
    ):
        self.grid_shape = grid_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.eps = eps
        self.discount = discount
        self.Q = np.zeros(
            (
                grid_shape[0] + 3,
                grid_shape[1] + 3,
                grid_shape[0] + 3,
                grid_shape[1] + 3,
                n_actions,
            )
        )

    def choose_action_train(self, state, observation=None):
        if random.random() < self.eps:
            action = random.choice(ACTIONS)
        else:
            q = self.Q[state]
            action = np.argmax(q)

        return action

    def choose_action_best(self, state):
        return ACTIONS[np.argmax(self.Q[state])]

    def learn(self, state, action, reward, next_state):
        self.Q[state][action] = (1 - self.learning_rate) * self.Q[state][
            action
        ] + self.learning_rate * (reward + self.discount * np.max(self.Q[next_state]))
