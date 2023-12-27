from weka.classifiers import Classifier

from src.learners.symbolic_learner.models.weka.base import WekaBasedModel


class C45WekaModel(WekaBasedModel):
    model = (Classifier, dict(classname="weka.classifiers.trees.J48"))
