import abc
from typing import Dict

from src.environments.models import BaseView


class BasePredictionModel(abc.ABC):

    @abc.abstractmethod
    def train(self, source: Dict[BaseView, Dict[int, int]]):
        pass

    @abc.abstractmethod
    def predict(self, state: BaseView) -> int:
        pass
