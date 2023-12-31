import math
from collections import defaultdict

from src.environments.envs.examples import (
    maze_7,
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
    maze_13, maze_3,
)
from src.java_interop_utils import safe_init_jvm, safe_stop_jvm
from src.learners.progressive_qlearner import ProgressiveQLearner
from src.learners.qlearner import QLearner
from src.learners.symbolic_learner.learner import SymbolicLearner
# from src.learners.symbolic_learner.models.weka.naive_bayes import NaiveBayesWekaModel
from src.learners.symbolic_learner.models.kNN import kNNModel
from src.runner.runner import Runner

if __name__ == "__main__":
    # Initialize the JVM for Weka.
    safe_init_jvm()

    model = kNNModel

    mazes = {
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

    # Write header to out.csv
    with open("out_prog.csv", "w") as f:
        f.write("maze,first_win_no_symbolic_learner,first_win_symbolic_learner,type\n")

    symbolic_learners = defaultdict(lambda: SymbolicLearner(model))

    for index, (name, maze) in enumerate(mazes.items()):
        print(f"Running ProgressiveQLearner on maze {name}.")
        print(dict(symbolic_learners))

        class X:
            first_win_symbolic_learner = None
            first_win_no_symbolic_learner = None

        x = X()

        def log_first_win_and_stop(env, event, x):
            if event.name == "first_win":
                x.first_win_symbolic_learner = event

        runner = Runner(
            enable_observation=True,
            maze=maze,
            agent_builder=lambda agent, grid_shape: ProgressiveQLearner(
                grid_shape,
                predict=symbolic_learners[agent].predict if index > 0 else None,
            ),
            n_agents=1,
            iterations=10_000,
            event_callback=lambda env, event: log_first_win_and_stop(env, event, x),
            max_steps=1_000,
        )
        runner.run()

        # Append data to out.csv
        with open("out_prog.csv", "a") as f:
            if not x.first_win_symbolic_learner:
                f.write(f"{name},0,0,progressive\n")
            else:
                f.write(
                    f'{name},{x.first_win_symbolic_learner.data.get("episodes")},{x.first_win_symbolic_learner.data.get("number_of_steps")},progressive\n'
                )

        runner.configure(
            convergence_count=math.inf,
            iterations=1_000,
            enable_observation=True,
            action_logger=lambda agent, state, action: symbolic_learners[agent].log(
                state, action
            ),
        )

        runner.run()

        def log_first_win_and_stop(env, event, x):
            print(event.data)
            if event.name == "first_win":
                x.first_win_no_symbolic_learner = event
                # Here we do not want to stop the environment, because we want to train the symbolic model.

        no_symbolic_learner_runner = Runner(
            enable_observation=False,
            maze=maze,
            agent_builder=lambda _, grid_shape: QLearner(grid_shape),
            n_agents=1,
            iterations=10_000,
            event_callback=lambda env, event: log_first_win_and_stop(env, event, x),
            max_steps=1_000,
        )

        no_symbolic_learner_runner.run()

        with open("out_prog.csv", "a") as f:
            if not x.first_win_no_symbolic_learner:
                f.write(f"{name},0,0,qlearner\n")
            else:
                f.write(
                    f'{name},{x.first_win_no_symbolic_learner.data.get("episodes")},{x.first_win_no_symbolic_learner.data.get("number_of_steps")},qlearner\n'
                )

        print(symbolic_learners)
        for agent in symbolic_learners:
            print("Training model for agent {0}.".format(agent))
            symbolic_learners[agent].train()
    safe_stop_jvm()
