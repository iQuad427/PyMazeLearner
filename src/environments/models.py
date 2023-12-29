import abc
from dataclasses import dataclass
from typing import List, Union

import numpy as np

from src.environments.observer.observation.base import BaseObservation


class Objects:
    WALL = 0
    EMPTY = 1
    MINOTAUR = "minotaur"
    EXIT = "exit"
    AGENT = "agent"


class BaseView(abc.ABC):
    @abc.abstractmethod
    def flatten(self) -> tuple:
        pass

    @abc.abstractmethod
    def __hash__(self):
        pass


@dataclass
class DefaultView(BaseView):
    walls_agent: tuple
    walls_minotaur: tuple
    distance_minotaur: int
    distance_exit: int
    direction_minotaur: tuple
    direction_exit: tuple

    def __hash__(self):
        return hash(
            (
                self.walls_agent,
                self.walls_minotaur,
                self.distance_minotaur,
                self.distance_exit,
                self.direction_minotaur,
                self.direction_exit,
            )
        )

    def flatten(self) -> tuple:
        return tuple(
            self.walls_agent
            + self.walls_minotaur
            + (self.distance_minotaur, self.distance_exit)
            + self.direction_minotaur
            + self.direction_exit
        )


class GenericView(BaseView):
    def __init__(self, observations: List[List[int]]):
        self.observations = observations

    def __hash__(self):
        return hash(
            tuple(
                [tuple(obs) for obs in self.observations]
            )
        )

    def flatten(self) -> tuple:
        return tuple(
            [x for obs in self.observations for x in obs]
        )

    def __eq__(self, other):
        instance_test = isinstance(other, GenericView)
        value_t = [all(np.array(self.observations[i]) == np.array(other.observations[i])) for i, obs in
                   enumerate(self.observations)]
        value_test = all(value_t)
        return value_test and instance_test


class Actions:
    STAY = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


ACTIONS = [Actions.STAY, Actions.UP, Actions.RIGHT, Actions.DOWN, Actions.LEFT]
