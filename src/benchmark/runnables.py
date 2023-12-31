import abc
import math
from collections import defaultdict

from src.learners.progressive_qlearner import ProgressiveQLearner
from src.learners.qlearner import QLearner
from src.learners.symbolic_learner.learner import SymbolicLearner
from src.runner.runner import Runner


class BaseRunnable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self, mazes, **kwargs):
        raise NotImplementedError


class RunnableQLearner(BaseRunnable):
    def __init__(self):
        self.event = None

    def run(self, mazes, **kwargs):
        for maze in mazes:
            print("Running QLearner on maze {0}.".format(maze))
            try:

                def log_first_win_and_stop(env, event):
                    if event.name == "first_win":
                        self.event = event
                        env.force_stop = True

                runner = Runner(
                    maze=mazes[maze],
                    agent_builder=lambda _, grid_shape: QLearner(grid_shape),
                    event_callback=log_first_win_and_stop,
                )

                runner.run()

                yield maze, self.event
                self.event = None
            except Exception as e:
                print(
                    "An error occurred while running the QLearner on maze {0}.".format(
                        maze
                    )
                )
                print(e)


class RunnableProgressiveQLearner(BaseRunnable):
    def __init__(self):
        self.event = None

    def run(self, mazes=None, model_factory=None, cumulative=False, **kwargs):
        symbolic_learners = defaultdict(lambda: SymbolicLearner(model_factory))

        for index, (name, maze) in enumerate(mazes.items()):
            try:

                def log_first_win_and_stop(env, event):
                    if event.name == "first_win":
                        self.event = event

                runner = Runner(
                    maze=maze,
                    enable_observation=True,
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

                yield name, self.event
                self.event = None

                runner.configure(
                    train=False,
                    convergence_count=math.inf,
                    iterations=1_000,
                    action_logger=lambda agent, state, action: symbolic_learners[
                        agent
                    ].log(state, action),
                )

                runner.run()

                for agent in symbolic_learners:
                    symbolic_learners[agent].train()

            except Exception as e:
                print(
                    "An error occurred while running the ProgressiveQLearner on maze {0}.".format(
                        maze
                    )
                )
                print(e)
