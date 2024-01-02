import math
from collections import defaultdict

from src.environments.envs.examples import maze_6, maze_7, maze_15, maze_9, maze_8, maze_1, maze_2, maze_3, maze_14
from src.environments.observer.observation.default import *
from src.java_interop_utils import safe_init_jvm, safe_stop_jvm
from src.learners.progressive_qlearner import ProgressiveQLearner
from src.learners.symbolic_learner.learner import SymbolicLearner
from src.learners.symbolic_learner.models.naive_bayes import NaiveBayesModel
# from src.learners.symbolic_learner.models.weka.naive_bayes import NaiveBayesWekaModel
# from src.learners.symbolic_learner.models.weka.part_dt import PDTWekaModel
from src.runner.runner import Runner
from src.environments.observer.generic import generic_default, GenericObserver

from src.learners.symbolic_learner.models.random_forest import RandomForestModel
from src.learners.symbolic_learner.models.neural_net import NeuralNetworkModel
from src.learners.symbolic_learner.models.kNN import kNNModel

if __name__ == "__main__":
    # Initialize the JVM for Weka.
    safe_init_jvm()
    observer = GenericObserver(
        [
            WallAgentObservation(),
            # WallMinoObservation(),
            # DistMinoObservation(),
            # DistExitObservation(),
            # DirMinoObservation(),
            # DirExitObservation()
        ]
    )
    model = RandomForestModel

    runner = Runner(
        enable_observation=False,
        # convergence_count=math.inf,
        maze=maze_14,
        agent_builder=lambda _, grid_shape: ProgressiveQLearner(grid_shape),
        n_agents=1,
        iterations=10_000,
        max_steps=math.inf,
        observer=observer,
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
        render_mode=None,
    )

    runner.run()

    # Train the symbolic models.
    for agent in symbolic_learners:
        symbolic_learners[agent].train()

    progressive_runner = Runner(
        maze=maze_14,
        agent_builder=lambda agent, grid_shape: ProgressiveQLearner(
            grid_shape, predict=symbolic_learners[agent].predict
        ),
        n_agents=1,
        iterations=10_000,
        max_steps=math.inf,
        # convergence_count=math.inf,
        render_mode=None,
        observer=observer
    )

    progressive_runner.run()

    progressive_runner.configure(
        train=False,
        convergence_count=math.inf,
        iterations=10,
        render_mode="human",
        sleep_time=0.2,
    )

    progressive_runner.run()

    safe_stop_jvm()
