from weka.classifiers import Classifier

from src.learners.symbolic_learner.models.weka.base import WekaBasedModel


class PDTWekaModel(WekaBasedModel):
    model = (Classifier, dict(classname="weka.classifiers.rules.PART"))
