from typing import Dict, Type

from src.environments.models import BaseView
from src.learners.symbolic_learner.models.base import BasePredictionModel


class AnisotropicWrapperModel(BasePredictionModel):
    @staticmethod
    def build_constructor(inner_model: Type[BasePredictionModel]):
        def constructor():
            return AnisotropicWrapperModel(inner_model())

        return constructor

    def __init__(self, inner_model: BasePredictionModel):
        self.inner_model = inner_model

    def train(self, source: Dict[BaseView, Dict[int, int]]):
        outer_source = {}

        for gv in source.keys():
            for oriented_view in gv.get_oriented_views():
                outer_source[oriented_view] = source[gv]

        self.inner_model.train(source)

    def predict(self, state: BaseView) -> int:
        return self.inner_model.predict(state)
