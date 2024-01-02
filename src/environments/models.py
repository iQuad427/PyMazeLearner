import abc
from dataclasses import dataclass
from typing import List

import numpy as np


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
    def names(self) -> tuple:
        pass

    @abc.abstractmethod
    def __eq__(self, other):
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

    def names(self) -> tuple:
        return (
            "walls_agent_1",
            "walls_agent_2",
            "walls_agent_3",
            "walls_agent_4",
            "walls_minotaur_1",
            "walls_minotaur_2",
            "walls_minotaur_3",
            "walls_minotaur_4",
            "distance_minotaur",
            "distance_exit",
            "direction_minotaur_1",
            "direction_minotaur_2",
            "direction_exit_1",
            "direction_exit_2",
        )

    def __eq__(self, other):
        return (
            self.walls_agent == other.walls_agent
            and self.walls_minotaur == other.walls_minotaur
            and self.distance_minotaur == other.distance_minotaur
            and self.distance_exit == other.distance_exit
            and self.direction_minotaur == other.direction_minotaur
            and self.direction_exit == other.direction_exit
        )


class GenericView(BaseView):
    def names(self) -> tuple:
        return tuple(
            # TODO: This is a hack. We should not have to do this.
            [f"obs_{i}" for i in range(len(self.observations))]
        )

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
