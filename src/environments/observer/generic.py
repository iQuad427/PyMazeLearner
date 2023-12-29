from typing import Dict, List

from src.environments.models import BaseView, GenericView
from src.environments.observer.base import BaseObserver
from src.environments.observer.observation.base import BaseObservation, GeneralObservation, AgentBasedObservation


class GenericObserver(BaseObserver):
    def __init__(self, observations: List[BaseObservation]):
        self._observations = observations

    def get_observations(self, env) -> Dict[str, BaseView]:
        observations = {}

        general_observations = [obs.get_observation(env) for obs in self._observations if
                                isinstance(obs, GeneralObservation)]

        for agent in env.agents:
            agent_observations = [obs.get_observation(env, agent) for obs in self._observations if
                                  isinstance(obs, AgentBasedObservation)]

            observations[agent] = GenericView(
                general_observations + agent_observations
            )

        return observations
