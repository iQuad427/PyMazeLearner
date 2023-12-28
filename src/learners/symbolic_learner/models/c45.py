from typing import Dict

from sklearn.tree import DecisionTreeClassifier

from src.environments.models import BaseView
from src.learners.symbolic_learner.models.base import BasePredictionModel


class C45Model(BasePredictionModel):
    def __init__(self):
        self.model = None

    def train(self, source: Dict[BaseView, Dict[int, int]]):
        """Trains a model using the given data."""
        features = [gv.flatten() for gv in source.keys()]
        # Todo: we use max to get the most common move, maybe not what rhe paper does
        labels = [max(source[gv], key=source[gv].get) for gv in source.keys()]

        model = DecisionTreeClassifier(criterion="entropy")  # C4.5 uses entropy
        model.fit(features, labels)

        self.model = model

    def predict(self, state: BaseView) -> int:
        return self.model.predict([state.flatten()])[0]
