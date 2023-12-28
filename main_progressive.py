import math
from collections import defaultdict

from src.environments.envs.examples import maze_6, maze_7
from src.java_interop_utils import safe_init_jvm, safe_stop_jvm
from src.learners.progressive_qlearner import ProgressiveQLearner
from src.learners.symbolic_learner.learner import SymbolicLearner
from src.learners.symbolic_learner.models.naive_bayes import NaiveBayesModel
from src.runner.runner import Runner

if __name__ == "__main__":
    # Initialize the JVM for Weka.
    safe_init_jvm()

    runner = Runner(
        enable_observation=False,
        maze=maze_7,
        agent_builder=lambda _, grid_shape: ProgressiveQLearner(grid_shape),
    )

    runner.run()

    symbolic_learners = defaultdict(lambda: SymbolicLearner(NaiveBayesModel))

    runner.configure(
        train=False,
        convergence_count=math.inf,
        iterations=1_000,
        action_logger=lambda agent, state, action: symbolic_learners[agent].log(
            state, action
        ),
    )

    runner.run()

    # Train the symbolic models.
    for agent in symbolic_learners:
        symbolic_learners[agent].train()

    progressive_runner = Runner(
        maze=maze_6,
        agent_builder=lambda agent, grid_shape: ProgressiveQLearner(
            grid_shape, predict=symbolic_learners[agent].predict
        ),
    )

    progressive_runner.run()

    progressive_runner.configure(
        render_mode="human",
        sleep_time=0.1,
    )

    progressive_runner.run()

    safe_stop_jvm()
