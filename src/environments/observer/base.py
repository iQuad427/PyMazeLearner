import abc
from typing import Dict

from src.environments.models import BaseView


class BaseObserver(abc.ABC):
    @abc.abstractmethod
    def get_observations(self, env) -> Dict[str, BaseView]:
        """
        This method should return a dictionary of observations for each agent.
        :param env: An instance of Environment from src.environments.env
        :return: A dictionary of observations for each agent.
        """
        raise NotImplementedError
