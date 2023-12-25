import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.environments.env import Environment
from src.environments.utils import grid_from_string
from src.environments.envs.examples import *
from src.learners.qlearner import QLearner

U, R, D, L = 1, 2, 3, 4

if __name__ == '__main__':
    grid, starting_pos = grid_from_string(maze_15)

    env = Environment(grid, starting_pos, max_steps=100_000)
    agents = {agent: QLearner(grid.shape) for agent in env.possible_agents}
    step_to_win = []
    for ep in tqdm(range(10000)):
        observations, infos = env.reset()
        while True:
            states = {agent: tuple(np.concatenate((infos[agent], infos['minautor']))) for agent in agents}
            actions = {agent: agents[agent].choose_action_train(states[agent]) for agent in states}
            # actions = {'player_A': solution.pop(0)}
            observations, rewards, terminations, truncations, infos = env.step(actions)

            new_states = {agent: tuple(np.concatenate((infos[agent], infos['minautor']))) for agent in agents}

            for agent in agents:
                agents[agent].learn(states[agent], actions[agent], rewards[agent], new_states[agent])
            if all(terminations.values()) or all(truncations.values()):
                for agent in agents:
                    if rewards[agent] == 1:
                        step_to_win.append(env.timestep)
                    else:
                        step_to_win.append(0)
                break

    for ep in tqdm(range(100)):
        observations, infos = env.reset()
        while True:
            states = {agent: tuple(np.concatenate((infos[agent], infos['minautor']))) for agent in agents}
            actions = {agent: agents[agent].choose_action_best(states[agent]) for agent in states}
            # actions = {'player_A': solution.pop(0)}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            new_states = {agent: tuple(np.concatenate((infos[agent], infos['minautor']))) for agent in agents}
            if all(terminations.values()) or all(truncations.values()):
                for agent in agents:
                    if rewards[agent] == 1:
                        step_to_win.append(env.timestep)
                    else:
                        step_to_win.append(0)
                break

    plt.plot(step_to_win)
    plt.show()