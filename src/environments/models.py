from dataclasses import dataclass


class Objects:
    WALL = 0
    EMPTY = 1
    MINOTAUR = "minotaur"
    EXIT = "exit"
    AGENT = "agent"


@dataclass(frozen=True, slots=True)
class GlobalView:
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


class Actions:
    STAY = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


ACTIONS = [Actions.STAY, Actions.UP, Actions.RIGHT, Actions.DOWN, Actions.LEFT]
