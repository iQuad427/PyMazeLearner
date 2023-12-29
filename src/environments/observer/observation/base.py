import abc
from typing import List


class BaseObservation(abc.ABC):
    @abc.abstractmethod
    def get_observation(self, env, agent) -> List[int]:
        """
        This method should return a tuple of the implemented observation for the specified agent
        :param env: An instance of Environment from src.environments.env
        :param agent: The name of the specified agent
        :return: A tuple of the observation for the specified agent
        """
        raise NotImplementedError


class AgentBasedObservation(BaseObservation):
    @abc.abstractmethod
    def get_observation(self, env, agent: str):
        raise NotImplementedError


class GeneralObservation(BaseObservation):
    @abc.abstractmethod
    def get_observation(self, env, agent: str = None):
        raise NotImplementedError
