import logging
from typing import Callable

import numpy as np
from tqdm import tqdm

from src.environments.env import Environment
from src.environments.models import Objects, GlobalView
from src.environments.utils import grid_from_string
from src.learners.qlearner import QLearner


class Runner:
    def __init__(
        self,
        maze: str,
        agent_builder: Callable[[str, tuple], QLearner],
        convergence_count=100,
        max_steps=200,
        iterations=10_000,
        render_mode=None,
        train=True,
        action_logger: Callable[[str, GlobalView, int], None] = None,
    ):
        self.convergence_count = convergence_count
        self.iterations = iterations
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.train = train
        self.grid, self.starting_pos = grid_from_string(maze)
        self.env = Environment(
            self.grid,
            self.starting_pos,
            max_steps=self.max_steps,
            render_mode=render_mode,
        )
        self.agents = {
            agent: agent_builder(agent, self.grid.shape)
            for agent in self.env.possible_agents
        }
        self.step_to_win = []
        self.action_logger = action_logger

    def configure(
        self,
        convergence_count=None,
        iterations=None,
        max_steps=None,
        render_mode=None,
        train=None,
        action_logger=None,
    ):
        if convergence_count is not None:
            self.convergence_count = convergence_count
        if iterations is not None:
            self.iterations = iterations
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
        )

    def run(self):
        for ep in tqdm(range(self.iterations), desc="Q-Learning"):
            # Reset the environment and get the initial observations.
            observations, infos = self.env.reset()

            self._run_once(self.env, ep, infos, observations)

    def _run_once(self, env, ep, infos, observations):
        while True:
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

            if all(terminations.values()) or all(truncations.values()):
                for agent in self.agents:
                    if rewards[agent] == 1:
                        self.step_to_win.append(env.timestep)
                        # Check if the agent has converged (i.e. has not improved in the last 100 episodes).
                        if len(self.step_to_win) > self.convergence_count:
                            if (
                                self.step_to_win[-self.convergence_count :]
                                == self.step_to_win[-1] * self.convergence_count
                            ):
                                logging.info(
                                    f"Agent {agent} has converged after {ep} episodes."
                                )
                    else:
                        self.step_to_win.append(0)
                break

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
