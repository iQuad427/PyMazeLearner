from typing import Dict

from sklearn.naive_bayes import GaussianNB

from src.environments.models import BaseView
from src.learners.symbolic_learner.models.base import BasePredictionModel


class NaiveBayesModel(BasePredictionModel):
    def __init__(self):
        self.model = None

    def train(self, source: Dict[BaseView, Dict[int, int]]):
        """ Trains a model using the given data. """
        features = [gv.flatten() for gv in source.keys()]
        labels = [max(source[gv], key=source[gv].get) for gv in source.keys()]

        model = GaussianNB()

        model.fit(features, labels)

        self.model = model

    def predict(self, state: BaseView) -> int:
        return self.model.predict([state.flatten()])[0]
