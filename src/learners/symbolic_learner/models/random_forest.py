from typing import Dict
from src.environments.models import BaseView
from src.learners.symbolic_learner.models.base import BasePredictionModel
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(BasePredictionModel):

        def __init__(self):
            self.model = None

        def train(self, source: Dict[BaseView, Dict[int, int]]):

            features = [gv.flatten() for gv in source.keys()]
            labels = [max(source[gv], key=source[gv].get) for gv in source.keys()]

            model = RandomForestClassifier()
            model.fit(features, labels)

            self.model = model

        def predict(self, state: BaseView) -> int:
            return self.model.predict([state.flatten()])[0]