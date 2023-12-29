from typing import Dict

import numpy as np

from src.environments.models import BaseView, Objects, DefaultView
from src.environments.observer.base import BaseObserver
from src.environments.utils import manhattan_distance, _star


class DefaultObserver(BaseObserver):
    def get_observations(self, env) -> Dict[str, BaseView]:
        walls_minotaur = np.array(
            _star(env.pos[Objects.MINOTAUR], env.transition_matrix, include_all=True)
            == Objects.WALL
        )  # can't go -> walls

        observations = {}

        for agent in env.agents:
            walls_agent = np.array(
                _star(env.pos[agent], env.transition_matrix, include_all=True)
                == Objects.WALL
            )

            distance_minotaur = manhattan_distance(
                env.pos[Objects.MINOTAUR], env.pos[agent]
            )
            distance_exit = manhattan_distance(env.pos[Objects.EXIT], env.pos[agent])

            # Return direction of minotaur and exit
            direction_minotaur = np.sign(env.pos[Objects.MINOTAUR] - env.pos[agent])
            direction_exit = np.sign(env.pos[Objects.EXIT] - env.pos[agent])

            observations[agent] = DefaultView(
                walls_agent=tuple(walls_agent),
                walls_minotaur=tuple(walls_minotaur),
                distance_minotaur=distance_minotaur,
                distance_exit=distance_exit,
                direction_minotaur=tuple(direction_minotaur),
                direction_exit=tuple(direction_exit),
            )

        return observations
