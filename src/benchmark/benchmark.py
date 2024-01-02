import json
import multiprocessing
import os
import uuid
from datetime import datetime
from typing import Dict, List

from src.benchmark.runnables import (
    RunnableQLearner,
    RunnableProgressiveQLearner,
)
from src.environments.envs.examples import (
    maze_1,
    maze_2,
    maze_3,
    maze_4,
    maze_5,
    maze_6,
    maze_7,
    maze_8,
    maze_9,
    maze_10,
    maze_11,
    maze_12,
    maze_13,
    maze_14,
    maze_15,
)
from src.java_interop_utils import safe_init_jvm, safe_stop_jvm
from src.learners.symbolic_learner.models.manual import ManualModel
from src.learners.symbolic_learner.models.weka.c45 import C45WekaModel
from src.learners.symbolic_learner.models.weka.naive_bayes import NaiveBayesWekaModel
from src.learners.symbolic_learner.models.weka.part_dt import PDTWekaModel

RUNNABLE_FACTORY = "runnable_factory"
MODEL_FACTORY = "model_factory"
CUMULATIVE = "cumulative"
CONVERGENCE_COUNT = "convergence_count"
BIAS = "bias"
USE_FIRST_MAZE = "use_first_maze"

# Create locket to prevent concurrent access to the output file
lock = multiprocessing.Lock()


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, type):
            return obj.__name__
        return json.JSONEncoder.default(self, obj)


def run_benchmark(runnable_params, mazes, output_file, runs):
    safe_init_jvm()

    runnable = runnable_params.pop(RUNNABLE_FACTORY)
    params = runnable_params

    for _ in range(runs):
        instance = str(uuid.uuid4())
        win = 0
        previous_maze = None

        for maze, event in runnable().run(mazes, **params):
            if previous_maze != maze:
                instance = str(uuid.uuid4())
                win = 0
                previous_maze = maze

            if event:
                win += 1

                lock.acquire()

                with open(output_file, "a") as f:
                    values = [
                        instance,
                        runnable.__name__,
                        runnable_params.get(MODEL_FACTORY).__name__
                        if runnable_params.get(MODEL_FACTORY)
                        else None,
                        runnable_params.get(CUMULATIVE, False),
                        runnable_params.get("bias", -0.2),
                        runnable_params.get("use_first_maze", False),
                        maze,
                        win,
                        event.data["number_of_steps"],
                        event.data["episodes"],
                        event.name == "first_win",
                        runnable_params.get(CONVERGENCE_COUNT, 200),
                        event.data["cumulative"],
                    ]

                    # Acquire the lock to prevent concurrent access to the output file
                    f.write(",".join(map(str, values)) + "\n")
                    # Release the lock

                lock.release()

    safe_stop_jvm()


def benchmark_runnables(
    mazes: Dict[str, str],
    runnables: List[Dict],
    runs=5,
):
    # Create the output directory if it does not exist.
    os.makedirs("out", exist_ok=True)

    name = f"benchmark_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_file = os.path.join("out", f"{name}.csv")
    output_file_config = os.path.join("out", f"{name}_config.csv")

    with open(output_file_config, "w") as f:
        f.write(
            json.dumps(
                {
                    "mazes": list(mazes.keys()),
                    "runnables": runnables,
                    "runs": runs,
                },
                indent=4,
                cls=CustomEncoder,
            )
        )

    with open(output_file, "w") as f:
        f.write(
            ",".join(
                [
                    "instance",
                    "runnable",
                    "model",
                    "cumulative",
                    "bias",
                    "use_first_maze",
                    "maze",
                    "win",
                    "steps",
                    "episodes",
                    "first_win",
                    "convergence_count",
                    "cumulative_reward",
                ]
            )
            + "\n"
        )

    processes = []

    for runnable_params in runnables:
        # Create a new process for each runnable
        p = multiprocessing.Process(
            target=run_benchmark,
            args=(
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
        "maze_14": maze_14,
        "maze_15": maze_15,
    }
    # Pour avoir dans l'article
    # Tous les WK
    # PDTWekaModel, C45WekaModel, NaiveBayesWekaModel, ManualModel
    # Tout ça cumulatif, convergence_count=1_000
    # BIAS = -0.2 et 0

    configs = [
        {
            RUNNABLE_FACTORY: RunnableQLearner,
            CONVERGENCE_COUNT: 1_000,
        }
    ]

    for bias in [-0.2, 0]:
        for cumulative in [True, False]:
            for convergence_count in [1_000]:
                for model in [
                    ManualModel,
                    PDTWekaModel,
                    C45WekaModel,
                    NaiveBayesWekaModel,
                ]:
                    for use_first_maze in [False]:
                        configs.append(
                            {
                                RUNNABLE_FACTORY: RunnableProgressiveQLearner,
                                MODEL_FACTORY: model,
                                CUMULATIVE: cumulative,
                                CONVERGENCE_COUNT: convergence_count,
                                BIAS: bias,
                                USE_FIRST_MAZE: use_first_maze,
                            }
                        )

    benchmark_runnables(
        _mazes,
        configs,
        runs=10,
    )
