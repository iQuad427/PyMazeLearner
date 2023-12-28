import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.environments.env import Environment
from src.environments.envs.examples import *
from src.environments.models import Objects
from src.environments.utils import grid_from_string
from src.learners.qlearner import QLearner

if __name__ == '__main__':
    grid, starting_pos = grid_from_string(maze_1)

    print(starting_pos)

    starting_pos_agent = starting_pos['player_A']

    starting_pos = {
        **starting_pos,
        'player_B': deepcopy(starting_pos_agent),
        'player_C': deepcopy(starting_pos_agent),
        'player_D': deepcopy(starting_pos_agent),
    }

    env = Environment(grid, starting_pos, max_steps=10_000)

    print(env.possible_agents)

    agents = {agent: QLearner(grid.shape) for agent in env.possible_agents}

    step_to_win = []
    for ep in tqdm(range(1_000)):
        observations, infos = env.reset()
        while True:
            states = {agent: tuple(np.concatenate((infos[agent], infos[Objects.MINOTAUR]))) for agent in agents}
            actions = {agent: agents[agent].choose_action_train(states[agent]) for agent in states}

            observations, rewards, terminations, truncations, infos = env.step(actions)

            new_states = {agent: tuple(np.concatenate((infos[agent], infos[Objects.MINOTAUR]))) for agent in agents}

            for agent in agents:
                agents[agent].learn(states[agent], actions[agent], rewards[agent], new_states[agent])

            if all(terminations.values()) or all(truncations.values()):
                for agent in agents:
                    if rewards[agent] == 1:
                        step_to_win.append(env.timestep)
                    else:
                        step_to_win.append(0)
                break

            # time.sleep(0.1)

    env = Environment(grid, starting_pos, max_steps=10_000, render_mode='human')

    for ep in tqdm(range(1)):
        observations, infos = env.reset()
        while True:
            states = {agent: tuple(np.concatenate((infos[agent], infos[Objects.MINOTAUR]))) for agent in agents}
            actions = {agent: agents[agent].choose_action_best(states[agent]) for agent in states}

            observations, rewards, terminations, truncations, infos = env.step(actions)

            new_states = {agent: tuple(np.concatenate((infos[agent], infos[Objects.MINOTAUR]))) for agent in agents}
            if all(terminations.values()) or all(truncations.values()):
                for agent in agents:
                    if rewards[agent] == 1:
                        step_to_win.append(env.timestep)
                    else:
                        step_to_win.append(0)
                break

            time.sleep(0.5)

    plt.plot(step_to_win)
    plt.show()
