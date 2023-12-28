from copy import deepcopy

import numpy as np

from src.environments.envs.examples import *
from src.environments.models import Objects


def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))


def grid_from_string(grid_str):
    s = grid_str.split('\n')
    grid = []
    starting_pos = {}
    for ri, r in enumerate(s):
        row = []
        for ci, c in enumerate(r):
            if c == "." or 'A' <= c <= 'Z' or 'a' <= c <= 'z':
                row.append([int(s[ri - 1][ci] != "_"), int(s[ri][ci + 1] != "|"), int(s[ri + 1][ci] != "_"),
                            int(s[ri][ci - 1] != "|")])
            if c == 'M':
                starting_pos[Objects.MINOTAUR] = np.array([ri // 2 - 1, ci // 2])
            elif 'A' <= c <= 'Z' or 'a' <= c <= 'z':
                # starting_pos[str('player_' + c)] = np.array([ri // 2 - 1, ci // 2])
                starting_pos[Objects.AGENT] = np.array([ri // 2 - 1, ci // 2])
            if c == '#':
                if ri % 2 == 0 and ci % 2 == 0:
                    if ci == 0:
                        starting_pos[Objects.EXIT] = np.array([ri // 2 - 1, ci - 1])
                    else:
                        starting_pos[Objects.EXIT] = np.array([ri // 2 - 1, ci // 2])
                elif ri % 2 == 1 and ci % 2 == 1:
                    if ri == 1:
                        starting_pos[Objects.EXIT] = np.array([ri // 2 - 1, ci // 2])
                    else:
                        starting_pos[Objects.EXIT] = np.array([ri // 2 + 1, ci // 2])
        if row:
            grid.append(row)
    return np.array(grid), starting_pos


def add_agents(starting_pos, n_agents):
    position = starting_pos[Objects.AGENT]

    for i in range(1, n_agents):
        starting_pos['player_' + chr(ord('A') + i)] = deepcopy(position)

    return starting_pos


# if __name__ == '__main__':
#     grid, starting_pos = grid_from_string(simple_maze)
#     print(grid.shape, starting_pos)
#     grid, starting_pos = grid_from_string(maze_1)
#     print(grid.shape, starting_pos)
#     grid, starting_pos = grid_from_string(maze_10)
#     print(grid.shape, starting_pos)
#     grid, starting_pos = grid_from_string(maze_12)
#     print(grid.shape, starting_pos)
#     grid, starting_pos = grid_from_string(maze_15)
#     print(grid.shape, starting_pos)
