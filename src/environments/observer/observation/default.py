import numpy as np

from src.environments.models import Objects
from src.environments.observer.observation.base import GeneralObservation, AgentBasedObservation
from src.environments.utils import _star, manhattan_distance


class WallMinoObservation(GeneralObservation):
    def get_observation(self, env, agent=None):
        return np.array(
            _star(env.pos[Objects.MINOTAUR], env.transition_matrix, include_all=True)
            == Objects.WALL
        )


class WallAgentObservation(AgentBasedObservation):
    def get_observation(self, env, agent):
        return np.array(
            _star(env.pos[agent], env.transition_matrix, include_all=True)
            == Objects.WALL
        )


class DistMinoObservation(AgentBasedObservation):
    def get_observation(self, env, agent):
        return manhattan_distance(env.pos[Objects.MINOTAUR], env.pos[agent])


class DistExitObservation(AgentBasedObservation):
    def get_observation(self, env, agent):
        return manhattan_distance(env.pos[Objects.EXIT], env)


class DirMinoObservation(AgentBasedObservation):
    def get_observation(self, env, agent):
        return np.sign(env.pos[Objects.MINOTAUR] - env.pos[agent])


class DirExitObservation(AgentBasedObservation):
    def get_observation(self, env, agent):
        return np.sign(env.pos[Objects.EXIT] - env.pos[agent])
