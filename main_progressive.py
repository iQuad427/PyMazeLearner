import math
from collections import defaultdict

from src.environments.envs.examples import maze_6, maze_7, maze_15, maze_9, maze_8, maze_1, maze_2
from src.java_interop_utils import safe_init_jvm, safe_stop_jvm
from src.learners.progressive_qlearner import ProgressiveQLearner
from src.learners.symbolic_learner.learner import SymbolicLearner
from src.learners.symbolic_learner.models.naive_bayes import NaiveBayesModel
# from src.learners.symbolic_learner.models.weka.naive_bayes import NaiveBayesWekaModel
# from src.learners.symbolic_learner.models.weka.part_dt import PDTWekaModel
from src.runner.runner import Runner

from src.learners.symbolic_learner.models.random_forest import RandomForestModel
from src.learners.symbolic_learner.models.neural_net import NeuralNetworkModel

if __name__ == "__main__":
    # Initialize the JVM for Weka.
    safe_init_jvm()

    runner = Runner(
        enable_observation=False,
        maze=maze_1,
        agent_builder=lambda _, grid_shape: ProgressiveQLearner(grid_shape),
        n_agents=1,
        iterations=1_000,
        max_steps=1_000,
    )
    runner.run()


    symbolic_learners = defaultdict(lambda: SymbolicLearner(NaiveBayesModel))
    #
    runner.configure(
        train=False,
        convergence_count=math.inf,
        iterations=1_000,
        enable_observation=True,
        action_logger=lambda agent, state, action: symbolic_learners[agent].log(
            state, action
        ),
    )
    print("Running ProgressiveQLearner on maze 7.")

    runner.run()

    symbolic_learners = dict(symbolic_learners)

    # Train the symbolic models.
    for agent in symbolic_learners:
        symbolic_learners[agent].train()
    #
    progressive_runner = Runner(
        maze=maze_2,
        agent_builder=lambda agent, grid_shape: ProgressiveQLearner(
            grid_shape, predict=symbolic_learners[agent].predict
        ),
    )

    progressive_runner.run()

    progressive_runner.configure(
        train=False,
        convergence_count=math.inf,
        render_mode="human",
        sleep_time=0.2,
    )

    progressive_runner.run()

    safe_stop_jvm()
