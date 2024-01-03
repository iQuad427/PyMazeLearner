import json
import multiprocessing
import os
from datetime import datetime
from typing import Dict, List

from src.benchmark.benchmark import RUNNABLE_FACTORY, CONVERGENCE_COUNT, MODEL_FACTORY, CUMULATIVE, BIAS, \
    USE_FIRST_MAZE, benchmark_runnables, run_benchmark, CustomEncoder
from src.benchmark.runnables import RunnableQLearner, RunnableProgressiveQLearner
from src.environments.envs.examples import *
from src.environments.observer.generic import GenericObserver
from src.environments.observer.observation.default import *
from src.learners.symbolic_learner.models.random_forest import RandomForestModel

OBSERVER = "observer"

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
                    "observation_used",
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

    configs = [
        {
            RUNNABLE_FACTORY: RunnableQLearner,
            CONVERGENCE_COUNT: 1000,
        }
    ]

    observations = [
        WallMinoObservation(),
        WallAgentObservation(),
        DistMinoObservation(),
        DistExitObservation(),
        DirMinoObservation(),
        DirExitObservation(),
    ]

    bias = 0
    cumulative = True
    convergence_count = 1000
    model = RandomForestModel
    use_first_maze = True
    for i in range(len(observations)):
        observer = GenericObserver(
            observations[:i] + observations[i+1:]
        )
        configs.append(
            {
                RUNNABLE_FACTORY: RunnableProgressiveQLearner,
                MODEL_FACTORY: model,
                CUMULATIVE: cumulative,
                CONVERGENCE_COUNT: convergence_count,
                BIAS: bias,
                USE_FIRST_MAZE: use_first_maze,
                OBSERVER: observer,
            }
        )

    benchmark_runnables(
        _mazes,
        configs,
        runs=2,
    )
