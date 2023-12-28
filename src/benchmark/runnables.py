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
        self.events = {}

    def run(self, mazes, **kwargs):
        for maze in mazes:
            try:

                def log_first_win_and_stop(env, event):
                    if event.name == "first_win":
                        self.events[maze] = event
                        env.force_stop = True

                runner = Runner(
                    maze=mazes[maze],
                    agent_builder=lambda _, grid_shape: QLearner(grid_shape),
                    event_callback=log_first_win_and_stop,
                )

                runner.run()
            except Exception as e:
                print(
                    "An error occurred while running the QLearner on maze {0}.".format(
                        maze
                    )
                )
                print(e)
        return self.events


class RunnableProgressiveQLearner(BaseRunnable):
    def __init__(self):
        self.events = {}

    def run(self, mazes=None, model_factory=None, cumulative=False, **kwargs):
        symbolic_learners = defaultdict(lambda: SymbolicLearner(model_factory))

        for index, (name, maze) in enumerate(mazes.items()):

            def log_first_win_and_stop(env, event):
                if event.name == "first_win":
                    self.events[name] = event
                    # Here we do not want to stop the environment, because we want to train the symbolic model.

            runner = Runner(
                maze=maze,
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

            runner.configure(
                train=False,
                convergence_count=math.inf,
                iterations=1_000,
                action_logger=lambda agent, state, action: symbolic_learners[agent].log(
                    state, action
                ),
            )

            runner.run()

            for agent in symbolic_learners:
                symbolic_learners[agent].train()

        return self.events
