import logging
import math
import time
from typing import Callable, Any

import numpy as np
from tqdm import tqdm

from src.environments.env import Environment
from src.environments.models import Objects, BaseView
from src.environments.observer.base import BaseObserver
from src.environments.utils import grid_from_string
from src.learners.qlearner import QLearner
from src.runner.event import Event, first_win


class Runner:
    def __init__(
        self,
        maze: str,
        agent_builder: Callable[[str, tuple], QLearner],
        convergence_count=300,
        max_steps=10_000,
        sleep_time=None,
        iterations=10_000,
        render_mode=None,
        train=True,
        action_logger: Callable[[str, BaseView, int], None] = None,
        observer: BaseObserver = None,
        event_callback: Callable[["Runner", Event], None] = None,
    ):
        self.convergence_count = convergence_count
        self.iterations = iterations
        self.max_steps = max_steps
        self.observer = observer
        self.render_mode = render_mode
        self.train = train
        self.sleep_time = sleep_time
        self.grid, self.starting_pos = grid_from_string(maze)
        self.env = Environment(
            self.grid,
            self.starting_pos,
            max_steps=self.max_steps,
            render_mode=render_mode,
            observer=self.observer,
        )
        self.agents = {
            agent: agent_builder(agent, self.grid.shape)
            for agent in self.env.possible_agents
        }
        self.step_to_win = []
        self.action_logger = action_logger

        self.did_already_win = False
        self.force_stop = False
        self.event_callback = event_callback

    def configure(
        self,
        convergence_count=None,
        iterations=None,
        max_steps=None,
        render_mode=None,
        train=None,
        sleep_time=None,
        action_logger=None,
    ):
        if convergence_count is not None:
            self.convergence_count = convergence_count
        if iterations is not None:
            self.iterations = iterations
        if sleep_time is not None:
            self.sleep_time = sleep_time
        if max_steps is not None:
            self.max_steps = max_steps
        if render_mode is not None:
            self.render_mode = render_mode
        if train is not None:
            self.train = train
        if action_logger is not None:
            self.action_logger = action_logger

        self.env = Environment(
            self.grid,
            self.starting_pos,
            max_steps=self.max_steps,
            render_mode=render_mode,
            observer=self.observer,
        )

    def run(self):
        for ep in tqdm(range(self.iterations), desc="Q-Learning"):
            if self.force_stop:
                logging.warning("Early interruption due to force stop")
                break
            # Reset the environment and get the initial observations.
            observations, infos = self.env.reset()

            if self._run_once(self.env, ep, infos, observations):
                logging.warning("Early interruption due to convergence")
                break

    def _run_once(self, env, ep, infos, observations):
        while True:
            if self.force_stop:
                logging.warning("Early interruption due to force stop")
                break

            # Get the states of all agents.
            states = self._get_states(infos)

            # Choose an action for each agent.
            actions = self._get_actions(states, observations)

            # Take a step in the environment.
            observations, rewards, terminations, truncations, infos = env.step(actions)

            for agent in observations:
                if observations.get(agent) and self.action_logger:
                    self.action_logger(agent, observations[agent], actions[agent])

            # Get the new states of all agents.
            new_states = self._get_states(infos)

            # Learn from the experience.
            self._learn(actions, new_states, rewards, states)

            if self.sleep_time:
                time.sleep(self.sleep_time)

            if all(terminations.values()) or all(truncations.values()):
                for agent in self.agents:
                    if rewards[agent] == 1:
                        if not self.did_already_win:
                            self.event_callback(self, first_win(ep, env.timestep))
                            self.did_already_win = True
                        self.step_to_win.append(env.timestep)
                    else:
                        self.step_to_win.append(math.inf)
                break

        return False

    def _learn(self, actions, new_states, rewards, states):
        for agent in self.agents:
            self.agents[agent].learn(
                states[agent], actions[agent], rewards[agent], new_states[agent]
            )

    def _get_actions(self, states, observations):
        actions = {
            agent: self.agents[agent].choose_action_train(
                states[agent], observations[agent]
            )
            if self.train
            else self.agents[agent].choose_action_best(states[agent])
            for agent in states
        }
        return actions

    def _get_states(self, infos):
        states = {
            agent: tuple(np.concatenate((infos[agent], infos[Objects.MINOTAUR])))
            for agent in self.agents
        }
        return states
