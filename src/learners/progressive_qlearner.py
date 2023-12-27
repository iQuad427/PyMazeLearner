import logging
from typing import Callable

from src.environments.models import GlobalView, ACTIONS
from src.learners.qlearner import QLearner


class ProgressiveQLearner(QLearner):
    def __init__(self, grid_shape, n_actions=5, learning_rate=0.3, eps=0.01, discount=0.9, bias=-0.2,
                 predict: Callable[[GlobalView], int] = None):
        super().__init__(grid_shape, n_actions, learning_rate, eps, discount)

        self.predict = predict

        if predict is None:
            logging.warning("No prediction function provided, standard QLearner will be used")

        self.explored = set()
        self.bias = bias

    def choose_action_train(self, state, observation=None):
        if self.predict is not  None and state not in self.explored and observation is not None:
            self.explored.add(state)
            action = self.predict(observation)

            for other_action in ACTIONS:
                if other_action != action:
                    self.Q[state][other_action] = self.bias

            return action

        return super().choose_action_train(state)

    def choose_action_best(self, state):
        return super().choose_action_best(state)

    def learn(self, state, action, reward, next_state):
        super().learn(state, action, reward, next_state)
