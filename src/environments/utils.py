import numpy as np

from src.environments.envs.examples import *
from src.environments.models import Objects


def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))
def _star(pos, data, include_all=False):
    return data[pos[0], pos[1], :] if include_all else data[pos[0], pos[1]]


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
                starting_pos[str('player_' + c)] = np.array([ri // 2 - 1, ci // 2])
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


if __name__ == '__main__':
    grid, starting_pos = grid_from_string(simple_maze)
    print(grid.shape, starting_pos)
    grid, starting_pos = grid_from_string(maze_1)
    print(grid.shape, starting_pos)
    grid, starting_pos = grid_from_string(maze_10)
    print(grid.shape, starting_pos)
    grid, starting_pos = grid_from_string(maze_12)
    print(grid.shape, starting_pos)
    grid, starting_pos = grid_from_string(maze_15)
    print(grid.shape, starting_pos)
