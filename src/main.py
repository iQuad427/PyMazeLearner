import numpy as np

from src.environments.env import Environment

if __name__ == '__main__':
    grid = np.array(
        [
            [
                [0, 1, 0, 0],
                [0, 1, 1, 1],
                [1, 0, 1, 1]
            ],
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [1, 0, 1, 0],
            ],
            [
                [1, 1, 0, 0],
                [0, 1, 0, 1],
                [1, 0, 0, 1]
            ]
        ]
    )

    starting_pos = {
        'player_0': np.array([0, 0]),
        'minautor': np.array([1, 0]),
        'exit': np.array([-1, 2])
    }

    env = Environment(grid, starting_pos, max_steps=1)

    observations, info = env.reset()

    print(env.pos)
    print(observations)

    # while env.agents:
    #     observations, rewards, terminations, truncations, infos = env.step({'player_0': 1})
    #     print(rewards)
    #     print(env.pos)

    observations, rewards, terminations, truncations, infos = env.step({'player_0': 0})
    print(rewards)
    print(env.pos)

    # observations, rewards, terminations, truncations, infos = env.step({'player_0': 1})
    # print(rewards)
    # print(env.pos)
    #
    # observations, rewards, terminations, truncations, infos = env.step({'player_0': 0})
    # print(rewards)
    # print(env.pos)

    print(terminations)
    print(truncations)

    print(observations)
