import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.environments.env import Environment
from src.environments.utils import grid_from_string
from src.environments.envs.examples import *
from src.learners.qlearner import QLearner

U, R, D, L = 1, 2, 3, 4

if __name__ == '__main__':
    grid, starting_pos = grid_from_string(maze_1)

    env = Environment(grid, starting_pos, max_steps=1000000)
    agents = {agent: QLearner(grid.shape) for agent in env.possible_agents}
    for _ in tqdm(range(100000)):
        observations, info = env.reset()
        while True:
            states = {agent: tuple(np.concatenate((info[agent], info['minautor']))) for agent in agents}
            actions = {agent: agents[agent].choose_action_train(states[agent]) for agent in states}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            new_states = {agent: tuple(np.concatenate((info[agent], info['minautor']))) for agent in agents}
            for agent in agents:
                agents[agent].learn(states[agent], actions[agent], rewards[agent], new_states[agent])
            if all(terminations.values()) or all(truncations.values()):
                if all(truncations.values()):
                    print("trunc")
                for agent in agents:
                    if rewards[agent] == 1:
                        exit(1)
                break
