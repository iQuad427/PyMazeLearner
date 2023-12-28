import abc
import math
from collections import defaultdict
from typing import Dict, Callable

from src.environments.envs.examples import (
    maze_1,
    maze_2,
    maze_4,
    maze_3,
    maze_5,
    maze_6,
    maze_8,
    maze_9,
    maze_10,
    maze_11,
    maze_12,
    maze_13,
    maze_14,
    maze_15,
)
from src.learners.progressive_qlearner import ProgressiveQLearner
from src.learners.qlearner import QLearner
from src.learners.symbolic_learner.learner import SymbolicLearner
from src.runner.runner import Runner


class BaseRunnable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self, mazes, **kwargs):
        raise NotImplementedError


class RunnableQLearner(BaseRunnable):
    def __init__(self):
        self.events = {}

    def run(self, mazes, **kwargs):
        for maze in mazes:
            def log_first_win_and_stop(env, event):
                if event.name == "first_win":
                    self.events[maze] = event
                    env.force_stop = True

            runner = Runner(
                maze=mazes[maze],
                agent_builder=lambda _, grid_shape: QLearner(grid_shape),
                event_callback=log_first_win_and_stop,
            )

            runner.run()

        return self.events


class RunnableProgressiveQLearner(BaseRunnable):
    def __init__(self):
        self.events = {}

    def run(self, mazes=None, model_factory=None, **kwargs):
        symbolic_learners = defaultdict(lambda: SymbolicLearner(model_factory))

        for index, maze in enumerate(mazes):

            def log_first_win_and_stop(env, event):
                if event.name == "first_win":
                    self.events[maze] = event
                    # Here we do not want to stop the environment, because we want to train the symbolic model.

            runner = Runner(
                maze=maze,
                agent_builder=lambda agent, grid_shape: ProgressiveQLearner(
                    grid_shape,
                    # We only want to use the symbolic model once it has been trained.
                    predict=symbolic_learners[agent].predict if index > 0 else None,
                ),
                event_callback=log_first_win_and_stop,
            )

            runner.run()

            runner.configure(
                train=False,
                convergence_count=math.inf,
                iterations=1_000,
                action_logger=lambda agent, state, action: symbolic_learners[agent].log(
                    state, action
                ),
            )

            runner.run()

            for agent in symbolic_learners:
                symbolic_learners[agent].train()

        return self.events


def baseline_benchmark(
    mazes: Dict[str, str],
    runnable_factory: Callable[[], BaseRunnable],
    number_of_benchmark_runs=5,
    runnable_kwargs=None,
) -> Dict[str, Dict[str, int]]:
    number_of_steps = defaultdict(list)
    number_of_episodes = defaultdict(list)

    for _ in range(number_of_benchmark_runs):
        events = runnable_factory().run(mazes, **(runnable_kwargs or {}))

        for maze, event in events.items():
            number_of_steps[maze].append(event.data["number_of_steps"])
            number_of_episodes[maze].append(event.data["episodes"])

    return number_of_steps, number_of_episodes


if __name__ == "__main__":
    baseline_benchmark(
        {
            "maze_1": maze_1,
            "maze_2": maze_2,
            "maze_3": maze_3,
            "maze_4": maze_4,
            "maze_5": maze_5,
            "maze_6": maze_6,
            "maze_8": maze_8,
            "maze_9": maze_9,
            "maze_10": maze_10,
            "maze_11": maze_11,
            "maze_12": maze_12,
            "maze_13": maze_13,
            "maze_14": maze_14,
            "maze_15": maze_15,
        },
        number_of_benchmark_runs=10,
        runnable_factory=RunnableQLearner,
    )
