from typing import Dict
from src.environments.models import BaseView
from src.learners.symbolic_learner.models.base import BasePredictionModel

from sklearn.neural_network import MLPClassifier


class NeuralNetworkModel(BasePredictionModel):

    def __init__(self):
        self.model = None


    def train(self, source: Dict[BaseView, Dict[int, int]]):

        X = [gv.flatten() for gv in source.keys()]
        Y = [max(source[gv], key=source[gv].get) for gv in source.keys()]

        model = MLPClassifier(
            hidden_layer_sizes=(100, 100),
            max_iter=100,
            activation='relu',
            solver='adam',
            # verbose=True,
        )

        model.fit(X, Y)
        self.model = model

    def predict(self, state: BaseView) -> int:
        return self.model.predict([state.flatten()])[0]

