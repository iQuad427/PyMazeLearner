from typing import Dict

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from src.environments.models import BaseView
from src.learners.symbolic_learner.models.base import BasePredictionModel


class NeuralNetwork2Model(BasePredictionModel):
    def __init__(self):
        self.model = None

    def train(self, source: Dict[BaseView, Dict[int, int]]):
        X = [gv.flatten() for gv in source.keys()]
        Y = [max(source[gv], key=source[gv].get) for gv in source.keys()]

        # First, reduce the dimensionality to 5 using PCA
        pca = PCA(n_components=5)

        # Then, define a 2 layer neural network
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50))

        # Create a pipeline that first applies PCA, then trains the neural network
        self.model = make_pipeline(pca, mlp)

        # Fit the model with the training data
        self.model.fit(X, Y)

    def predict(self, state: BaseView) -> int:
        return self.model.predict([state.flatten()])[0]
