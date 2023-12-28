import os
from collections import defaultdict
from typing import Dict, Callable

from src.benchmark.runnables import (
    BaseRunnable,
    RunnableQLearner,
    RunnableProgressiveQLearner,
)
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
from src.learners.symbolic_learner.models.c45 import C45Model
from src.learners.symbolic_learner.models.naive_bayes import NaiveBayesModel


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
            events = runnable_factory().run(
                mazes, **(runnables_kwargs or {}).get(name, {})
            )

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
