import random
from typing import Dict

from src.environments.models import BaseView, ACTIONS, DefaultView, Actions
from src.learners.symbolic_learner.models.base import BasePredictionModel


class ManualModel(BasePredictionModel):
    def __init__(self):
        pass

    def train(self, source: Dict[BaseView, Dict[int, int]]):
        pass

    def predict(self, state: DefaultView) -> int:
        possible_actions = ACTIONS[1:]

        if state.walls_agent[0]:
            possible_actions.remove(Actions.UP)
        if state.walls_agent[1]:
            possible_actions.remove(Actions.RIGHT)
        if state.walls_agent[2]:
            possible_actions.remove(Actions.DOWN)
        if state.walls_agent[3]:
            possible_actions.remove(Actions.LEFT)

        return random.choice(possible_actions)


