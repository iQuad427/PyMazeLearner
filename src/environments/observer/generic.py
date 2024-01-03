from typing import Dict, List

from src.environments.env import Environment
from src.environments.models import BaseView, GenericView
from src.environments.observer.base import BaseObserver
from src.environments.observer.observation.base import BaseObservation, GeneralObservation, AgentBasedObservation
from src.environments.observer.observation.default import *
from src.environments.utils import grid_from_string, add_agents
from src.environments.envs.examples import maze_1


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

    def serialize(self):
        return "".join(self._observations.__class__.__name__)


generic_default = GenericObserver(
    [
        WallAgentObservation(),
        WallMinoObservation(),
        DistMinoObservation(),
        DistExitObservation(),
        DirMinoObservation(),
        DirExitObservation()
    ]
)


if __name__ == "__main__":
    observations = [WallMinoObservation(), WallMinoObservation(), DistMinoObservation(), DistMinoObservation(), DirMinoObservation(), DirExitObservation()]
    observer = GenericObserver(observations)
    grid, starting_pos = grid_from_string(maze_1)
    state = add_agents(starting_pos, 1)
    env = Environment(
        grid,
        starting_pos,
        enable_observation=True,
        observer=observer
    )
    obs, info = env.reset()
    print(obs['player_A'], obs['player_A'].__hash__())
    obs2, info2 = env.reset()
    print(obs2['player_A'], obs2['player_A'].__hash__())
