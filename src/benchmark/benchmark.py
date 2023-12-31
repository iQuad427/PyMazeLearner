import multiprocessing
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
    maze_5,
    maze_6,
    maze_8,
    maze_9,
    maze_10,
    maze_11,
    maze_12,
    maze_13,
    maze_3, maze_7,
)
from src.java_interop_utils import safe_init_jvm, safe_stop_jvm
from src.learners.symbolic_learner.models.naive_bayes import NaiveBayesModel
from src.learners.symbolic_learner.models.random_forest import RandomForestModel
from src.learners.symbolic_learner.models.weka.c45 import C45WekaModel
from src.learners.symbolic_learner.models.weka.naive_bayes import NaiveBayesWekaModel

RUNNABLE_FACTORY = "runnable_factory"
MODEL_FACTORY = "model_factory"
CUMLATIVE = "cumulative"


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


def run_benchmark(runnable_name, runnable_params, mazes, output_file, runs):
    safe_init_jvm()

    print("Running {0}.".format(runnable_name))
    runnable = runnable_params.pop(RUNNABLE_FACTORY)
    params = runnable_params

    for _ in range(runs):
        for maze, event in runnable().run(mazes, **params):
            if event:
                with open(output_file, "a") as f:
                    f.write(
                        "{0},{1},{2},{3}\n".format(
                            runnable_name,
                            maze,
                            event.data["number_of_steps"],
                            event.data["episodes"],
                        )
                    )

    safe_stop_jvm()

def benchmark_runnables(
    mazes: Dict[str, str],
    runnables: Dict[str, Dict],
    runs=5,
    output_file=None,
):
    output_file = output_file or "out.csv"
    with open(output_file, "w") as f:
        f.write("model,maze,steps,episodes\n")

    processes = []

    for name, runnable_params in runnables.items():
        # Create a new process for each runnable
        p = multiprocessing.Process(
            target=run_benchmark,
            args=(
                name,
                runnable_params,
                mazes,
                output_file,
                runs,
            ),
        )
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()


if __name__ == "__main__":

    _mazes = {
        "maze_1": maze_1,
        "maze_2": maze_2,
        "maze_3": maze_3,
        "maze_4": maze_4,
        "maze_5": maze_5,
        "maze_6": maze_6,
        "maze_7": maze_7,
        "maze_8": maze_8,
        "maze_9": maze_9,
        "maze_10": maze_10,
        "maze_11": maze_11,
        "maze_12": maze_12,
        "maze_13": maze_13,
    }

    benchmark_runnables(
        _mazes,
        {
            "QLearner": {
                RUNNABLE_FACTORY: RunnableQLearner,
            },
            "ProgressiveQLearner - C45Weka": {
                RUNNABLE_FACTORY: RunnableProgressiveQLearner,
                MODEL_FACTORY: C45WekaModel,
            },
            "ProgressiveQLearner - NaiveBayesWekaModel": {
                RUNNABLE_FACTORY: RunnableProgressiveQLearner,
                MODEL_FACTORY: NaiveBayesWekaModel,
            },
            "ProgressiveQLearner - RandomForestModel": {
                RUNNABLE_FACTORY: RunnableProgressiveQLearner,
                MODEL_FACTORY: RandomForestModel,
            },
            "ProgressiveQLearner - C45Weka - Cumulative": {
                RUNNABLE_FACTORY: RunnableProgressiveQLearner,
                MODEL_FACTORY: C45WekaModel,
                CUMLATIVE: True,
            },
            "ProgressiveQLearner - NaiveBayesWekaModel - Cumulative": {
                RUNNABLE_FACTORY: RunnableProgressiveQLearner,
                MODEL_FACTORY: NaiveBayesWekaModel,
                CUMLATIVE: True,
            },
            "ProgressiveQLearner - NaiveBayesModel - Cumulative": {
                RUNNABLE_FACTORY: RunnableProgressiveQLearner,
                MODEL_FACTORY: NaiveBayesModel,
                CUMLATIVE: True,
            },
        },
        runs=5,
    )


