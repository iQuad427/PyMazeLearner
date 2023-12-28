import abc
import math
import os
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
from src.learners.symbolic_learner.models.c45 import C45Model
from src.learners.symbolic_learner.models.naive_bayes import NaiveBayesModel
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
            try:

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
            except Exception as e:
                print(
                    "An error occurred while running the QLearner on maze {0}.".format(
                        maze
                    )
                )
                print(e)
        return self.events


class RunnableProgressiveQLearner(BaseRunnable):
    def __init__(self):
        self.events = {}

    def run(self, mazes=None, model_factory=None,cumulative=False, **kwargs):
        symbolic_learners = defaultdict(lambda: SymbolicLearner(model_factory))

        for index, (name, maze) in enumerate(mazes.items()):

            def log_first_win_and_stop(env, event):
                if event.name == "first_win":
                    self.events[name] = event
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

            if not cumulative:
                symbolic_learners.clear()

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


def retrieve_number_of_steps_and_episodes_per_maze(
    mazes: Dict[str, str],
    runnable_factory: Callable[[], BaseRunnable],
    runs=5,
    runnable_kwargs=None,
) -> (Dict[str, list], Dict[str, list]):
    number_of_steps = defaultdict(list)
    number_of_episodes = defaultdict(list)

    for _ in range(runs):
        events = runnable_factory().run(mazes, **(runnable_kwargs or {}))

        for maze, event in events.items():
            number_of_steps[maze].append(event.data["number_of_steps"])
            number_of_episodes[maze].append(event.data["episodes"])

    return number_of_steps, number_of_episodes


def benchmark_runnables(
    mazes: Dict[str, str],
    runnables: Dict[str, Callable[[], BaseRunnable]],
    runnables_kwargs: Dict[str, Dict] = None,
    runs=5,
):
    number_of_steps = defaultdict(lambda: defaultdict(list))
    number_of_episodes = defaultdict(lambda: defaultdict(list))

    for _ in range(runs):
        for name, runnable_factory in runnables.items():
            events = runnable_factory().run(mazes, **(runnables_kwargs or {}).get(name, {}))

            for maze, event in events.items():
                number_of_steps[name][maze].append(event.data["number_of_steps"])
                number_of_episodes[name][maze].append(event.data["episodes"])

    # Beautiful print
    print("Model\tMaze\tSteps\tEpisodes")
    for model, mazes in number_of_steps.items():
        for maze, steps in mazes.items():
            print(
                "{0}\t{1}\t{2}\t{3}".format(
                    model,
                    maze,
                    sum(steps) / len(steps),
                    sum(number_of_episodes[model][maze])
                    / len(number_of_episodes[model][maze]),
                )
            )




if __name__ == "__main__":
    supp = {
        "maze_8": maze_8,
        "maze_9": maze_9,
        "maze_10": maze_10,
        "maze_11": maze_11,
        "maze_12": maze_12,
        "maze_13": maze_13,
        "maze_14": maze_14,
        "maze_3": maze_3,
        "maze_15": maze_15,
        "maze_5": maze_5,
        "maze_6": maze_6,
        "maze_4": maze_4,
    }

    _mazes = {
        "maze_1": maze_1,
        "maze_2": maze_2,
    }

    if os.environ.get("supp"):
        _mazes.update(supp)

    benchmark_runnables(
        _mazes,
        {
            "QLearner": RunnableQLearner,
            "ProgressiveQLearner - C45": RunnableProgressiveQLearner,
            "ProgressiveQLearner - Naive bayes": RunnableProgressiveQLearner,
            "ProgressiveQLearner - C45 - Cumulative": RunnableProgressiveQLearner,
            "ProgressiveQLearner - Naive bayes - Cumulative": RunnableProgressiveQLearner,
        },
        runnables_kwargs={
            "ProgressiveQLearner - C45": {"model_factory": C45Model},
            "ProgressiveQLearner - Naive bayes": {"model_factory": NaiveBayesModel},
            "ProgressiveQLearner - C45 - Cumulative": {
                "model_factory": C45Model,
                "cumulative": True,
            },
            "ProgressiveQLearner - Naive bayes - Cumulative": {
                "model_factory": NaiveBayesModel,
                "cumulative": True,
            },
        },
        runs=1,
    )