import abc
from typing import Dict

from src.environments.models import GlobalView


class BasePredictionModel(abc.ABC):

    @abc.abstractmethod
    def train(self, source: Dict[GlobalView, Dict[int, int]]):
        pass

    @abc.abstractmethod
    def predict(self, state: GlobalView) -> int:
        pass
