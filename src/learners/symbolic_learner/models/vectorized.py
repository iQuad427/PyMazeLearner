from typing import Dict

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.environments.models import BaseView
from src.learners.symbolic_learner.models.base import BasePredictionModel


class VectorizedModel(BasePredictionModel):
    def __init__(self):
        self.scaler = None
        self.vectors = []
        self.labels = []

    def train(self, source: Dict[BaseView, Dict[int, int]]):
        features = [gv.flatten() for gv in source.keys()]
        labels = [max(source[gv], key=source[gv].get) for gv in source.keys()]

        # Normalize features
        features = np.array(features)
        # Use scaler to normalize features
        scaler = StandardScaler()

        # Fit on training set only.
        scaler.fit(features)
        self.scaler = scaler

        self.vectors = scaler.transform(features)
        self.labels = labels

    def predict(self, state: BaseView) -> int:
        return self.labels[
            np.argmin(
                np.linalg.norm(self.vectors - self.scaler.transform([state.flatten()]), axis=1)
            )
        ]




