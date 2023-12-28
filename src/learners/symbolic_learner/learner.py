from collections import defaultdict
from typing import Dict

from src.environments.models import BaseView
from src.learners.symbolic_learner.models.base import BasePredictionModel


class SymbolicLearner:
    """
    A learner that learns a symbolic model of the environment.
    The symbolic here means that this learner tries to capture global
    information about the environment, rather than learning a model for
    each state.

    This class is highly coupled but could be refactored to be more generic if
    needed.
    """
    def __init__(self, model_constructor):
        self.model_constructor = model_constructor
        self.history: Dict[BaseView, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.model: BasePredictionModel = self.model_constructor()
        self.did_train = False

    def log(self, state: BaseView, action: int):
        self.history[state][action] += 1

    def clear(self):
        self.did_train = False
        self.history = defaultdict(lambda: defaultdict(int))
        self.model = self.model_constructor()

    def train(self):
        self.did_train = True
        self.model = self.model_constructor()
        self.model.train(self.history)

    def predict(self, state: BaseView) -> int:
        assert self.did_train, "Must train before predicting"
        return self.model.predict(state)
