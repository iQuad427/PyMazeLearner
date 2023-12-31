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
    maze_13, maze_3, maze_14, maze_15,
)
from src.java_interop_utils import safe_init_jvm, safe_stop_jvm
from src.learners.progressive_qlearner import ProgressiveQLearner
from src.learners.symbolic_learner.learner import SymbolicLearner
from src.learners.symbolic_learner.models.weka.c45 import C45WekaModel
from src.runner.runner import Runner

if __name__ == "__main__":
    # Initialize the JVM for Weka.
    safe_init_jvm()

    model = C45WekaModel

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
        "maze_14": maze_14,
        "maze_15": maze_15
    }

    # Write header to out.csv
    with open('out.csv', 'w') as f:
        f.write('maze,episodes,steps\n')

    for name, maze in mazes.items():
        class X:
            first_win_no_symbolic_learner = None
            first_win_symbolic_learner = None
        x = X()

        def log_first_win_and_stop(env, event, x):
            if event.name == "first_win":
                x.first_win_no_symbolic_learner = event

        runner = Runner(
            enable_observation=False,
            maze=maze,
            agent_builder=lambda _, grid_shape: ProgressiveQLearner(grid_shape),
            n_agents=1,
            iterations=10_000,
            event_callback=lambda env, event: log_first_win_and_stop(env, event, x),
            max_steps=1_000,
        )
        runner.run()

        symbolic_learners = defaultdict(lambda: SymbolicLearner(model))
        runner.configure(
            train=False,
            convergence_count=math.inf,
            iterations=1_000,
            enable_observation=True,
            action_logger=lambda agent, state, action: symbolic_learners[agent].log(
                state, action
            ),
        )

        print(f"Running ProgressiveQLearner on maze {name}.")

        runner.run()

        symbolic_learners = dict(symbolic_learners)

        # Train the symbolic models.
        for agent in symbolic_learners:
            symbolic_learners[agent].train()

        def log_first_win_and_stop(env, event, x):
            if event.name == "first_win":
                x.first_win_symbolic_learner = event
                # Here we do not want to stop the environment, because we want to train the symbolic model.

        progressive_runner = Runner(
            maze=maze,
            agent_builder=lambda agent, grid_shape: ProgressiveQLearner(
                grid_shape, predict=symbolic_learners[agent].predict
            ),
            enable_observation=True,
            event_callback=lambda env, event: log_first_win_and_stop(env, event, x),
            iterations=10_000,
            max_steps=1_000,
        )

        progressive_runner.run()

        progressive_runner.configure(
            train=False,
            convergence_count=math.inf,
            render_mode="human",
            iterations=1,
            max_steps=1,
            sleep_time=0.2,
        )

        progressive_runner.run()

        # Append data to out.csv
        with open('out.csv', 'a') as f:
            if not x.first_win_no_symbolic_learner:
                f.write(f'{name},0,0\n')
            else:
                f.write(f'{name},{x.first_win_no_symbolic_learner.data.get("episodes")},{x.first_win_no_symbolic_learner.data.get("number_of_steps")}\n')

            if not x.first_win_symbolic_learner:
                f.write(f'{name},0,0\n')
            else:
                f.write(f'{name},{x.first_win_symbolic_learner.data.get("episodes")},{x.first_win_symbolic_learner.data.get("number_of_steps")}\n')

    safe_stop_jvm()
