from abc import ABC
from typing import Dict, Tuple, Any

from weka.core.dataset import Instances, Attribute, Instance

from src.environments.models import BaseView, ACTIONS
from src.learners.symbolic_learner.models.base import BasePredictionModel


class WekaBasedModel(BasePredictionModel, ABC):
    model: Tuple[Any, Dict] = None

    def __init__(self):
        self.dataset = None

    @staticmethod
    def _convert_to_weka_dataset(features, labels, names=None):
        """Converts the given data to a Weka dataset."""

        treated_features = [list(map(int, gv)) for gv in features]
        treated_labels = [str(label) for label in labels]


        print(names)

        attributes = [
            Attribute.create_numeric(names[i] if names else f"feature_{i}")
            for i in range(len(features[0]))
        ]


        attributes += [Attribute.create_nominal("label", list(map(str, ACTIONS)))]

        dataset = Instances.create_instances(
            "source",
            attributes,
            capacity=len(features),
        )

        for feature, label in zip(treated_features, treated_labels):
            values = feature + [label]
            instance = Instance.create_instance(values)
            dataset.add_instance(instance)

        dataset.class_is_last()

        return dataset

    def train(self, source: Dict[BaseView, Dict[int, int]]):
        """Trains a model using the given data."""
        assert self.model is not None, "Model generator must be defined"

        keys = list(source.keys())
        features = [gv.flatten() for gv in keys]
        labels = [max(source[gv], key=source[gv].get) for gv in keys]

        dataset = self._convert_to_weka_dataset(features, labels, names=keys[0].names())
        model = self.model[0](**self.model[1])

        model.build_classifier(dataset)

        self.dataset = dataset
        self.model = model

    def predict(self, state: BaseView) -> int:
        # Convert the state to a Weka instance.
        instance = Instance.create_instance([int(i) for i in state.flatten()])
        instance.dataset = self.dataset
        return int(self.model.classify_instance(instance))
