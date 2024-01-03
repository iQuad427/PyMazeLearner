import numpy as np

from src.environments.env import Environment
from src.environments.envs.examples import *
from src.environments.models import Objects
from src.environments.observer.generic import GenericObserver
from src.environments.observer.observation.base import AgentBasedObservation
from src.environments.observer.observation.default import WallMinoObservation
from src.environments.utils import _star, grid_from_string, add_agents


class LOSMinoAgentObservation(AgentBasedObservation):
    """
    Tells if the Minotaur is able to move toward the agent
    Type: Boolean
    """

    def get_observation(self, env, agent):
        # Direction from Mino to agent
        direction = np.sign(env.pos[agent] - env.pos[Objects.MINOTAUR])

        # Walls
        wallsMino = np.array(
            _star(env.pos[Objects.MINOTAUR], env.transition_matrix, include_all=True)
            == Objects.WALL
        )
        move_x = (direction[0] == -1 and not wallsMino[0]) or (direction[0] == 1 and not wallsMino[2])
        move_y = (direction[1] == -1 and not wallsMino[3]) or (direction[1] == 1 and not wallsMino[1])

        return [move_x or move_y]


class MaybeWaitObservation(AgentBasedObservation):
    """
    When the Minotaur is aligned with the agent and a wall is at any point between them
    the agent can wait for the Minotaur to reach the wall
    """

    def get_observation(self, env, agent):
        # Direction from Mino to agent
        direction = np.sign(env.pos[agent] - env.pos[Objects.MINOTAUR])

        if direction[1] == 0:
            dir_x = 2 if direction[0] == 1 else 0
            walls = any([_star(env.pos[Objects.MINOTAUR] + np.array([i, 0]), env.transition_matrix, include_all=True)[
                             dir_x] == 0
                         for i in range(0, (env.pos[agent] - env.pos[Objects.MINOTAUR])[0], direction[0])])

        elif direction[0] == 0:
            dir_y = 1 if direction[1] == 1 else 3
            walls = any([_star(env.pos[Objects.MINOTAUR] + np.array([0, i]), env.transition_matrix,
                               include_all=True)[dir_y] == 0
                         for i in range(0, (env.pos[agent] - env.pos[Objects.MINOTAUR])[1], direction[1])])

        else:
            walls = False

        return [walls]


class HorizontalMoveIsBlockingObservation(AgentBasedObservation):
    """
    Moving horizontally can block the Mino if a wall is between agent and minotaur
    """

    def get_observation(self, env, agent):
        # Direction from Mino to agent
        direction = np.sign(env.pos[agent] - env.pos[Objects.MINOTAUR])

        if direction[1] == 0:
            dir_x = 2 if direction[0] == 1 else 0
            print(dir_x)
            print(_star(env.pos[Objects.MINOTAUR], env.transition_matrix, include_all=True))
            if _star(env.pos[Objects.MINOTAUR], env.transition_matrix, include_all=True)[3] == 1:
                print("Here")
                left = (_star(env.pos[Objects.MINOTAUR] + np.array([0, -1]), env.transition_matrix, include_all=True)[
                            dir_x] == 0
                        or _star(env.pos[Objects.MINOTAUR] + np.array([direction[0], -1]), env.transition_matrix,
                                 include_all=True)[dir_x] == 0)
            else:
                print("Here 2")
                left = (_star(env.pos[Objects.MINOTAUR] + np.array([direction[0], -1]), env.transition_matrix,
                              include_all=True)[dir_x] == 0
                        and _star(env.pos[Objects.MINOTAUR] + np.array([direction[0], 0]), env.transition_matrix,
                                  include_all=True)[3] == 1)

            if _star(env.pos[Objects.MINOTAUR], env.transition_matrix, include_all=True)[1] == 1:
                right = (_star(env.pos[Objects.MINOTAUR] + np.array([0, 1]), env.transition_matrix,
                               include_all=True)[dir_x] == 0
                         or _star(env.pos[Objects.MINOTAUR] + np.array([direction[0], 1]), env.transition_matrix,
                                  include_all=True)[dir_x] == 0)
            else:
                right = (_star(env.pos[Objects.MINOTAUR] + np.array([direction[0], 1]), env.transition_matrix,
                               include_all=True)[dir_x] == 0
                         and _star(env.pos[Objects.MINOTAUR] + np.array([direction[0], 0]), env.transition_matrix,
                                   include_all=True)[1] == 1)
            return [left, right]

        return [False, False]


if __name__ == "__main__":
    """Tests"""

    observer = GenericObserver(
        [
            LOSMinoAgentObservation()
        ]
    )

    mazes = (
        """
+_+_+#+
|A+.+.|
+_+ + +
|M|.|.|
+ +_+ +
|.+.+.|
+_+_+_+
        """,
        """
+_+_+#+
|A+.+.|
+ + + +
|M|.|.|
+ +_+ +
|.+.+.|
+_+_+_+
        """,
        """
+_+_+#+
|A+.+.|
+ + + +
|.|.|.|
+ +_+ +
|.+M+.|
+_+_+_+
        """,
        """
+_+_+#+
|A+.+.|
+ + + +
|.|.|.|
+ +_+ +
|.|M+.|
+_+_+_+
        """,
    )
    answers = (False, True, True, False)
    for i in range(len(mazes)):
        grid, starting_pos = grid_from_string(mazes[i])
        add_agents(starting_pos, 1)
        env = Environment(
            grid,
            starting_pos,
            enable_observation=True,
            max_steps=1000,
            render_mode='human',
            observer=observer,
        )

        obs, infos = env.reset()
        assert obs['player_A'].flatten()[0] == answers[i]




    observer = GenericObserver(
        [
            MaybeWaitObservation()
        ]
    )

    mazes = (
        """
+_+_+#+
|A+.+.|
+_+ + +
|M|.|.|
+ +_+ +
|.+.+.|
+_+_+_+
        """,
        """
+_+_+#+
|A+.+.|
+ + + +
|M|.|.|
+ +_+ +
|.+.+.|
+_+_+_+
        """,
        """
+_+_+#+
|A+.+.|
+_+ + +
|.|.|.|
+ +_+ +
|.+M+.|
+_+_+_+
        """,
        """
+_+_+#+
|A+.+.|
+_+ + +
|.|.|.|
+ +_+ +
|M|.+.|
+_+_+_+
        """,
        """
+_+_+#+
|A+.+M|
+_+ + +
|.|.|.|
+ +_+ +
|.|.+.|
+_+_+_+
        """,
        """
+_+_+#+
|A+.|M|
+_+ + +
|.|.|.|
+ +_+ +
|.|.+.|
+_+_+_+
        """,
    )
    answers = (True, False, False, True, False, True)
    for i in range(len(mazes)):
        grid, starting_pos = grid_from_string(mazes[i])
        add_agents(starting_pos, 1)
        env = Environment(
            grid,
            starting_pos,
            enable_observation=True,
            max_steps=1000,
            render_mode='human',
            observer=observer,
        )

        obs, infos = env.reset()
        assert obs['player_A'].flatten()[0] == answers[i]




    observer = GenericObserver(
        [
            HorizontalMoveIsBlockingObservation()
        ]
    )

    mazes = (
        """
+_+_+#+
|.+A+.|
+_+ + +
|.+M|.|
+ +_+ +
|.+.+.|
+_+_+_+
        """,
        """
+_+_+#+
|A+.+.|
+ +_+ +
|M+.|.|
+ +_+ +
|.+.+.|
+_+_+_+
        """,
        """
+_+_+#+
|A+.+.|
+ +_+ +
|.+.|.|
+ +_+ +
|M+.+.|
+_+_+_+
        """,
        """
+_+_+#+
|A+.+M|
+_+ + +
|.|.|.|
+ +_+ +
|.|.+.|
+_+_+_+
        """,
    )
    answers = ((True, False), (False, True), (False, True), (False, False),)
    for i in range(2):
        print(i)
        grid, starting_pos = grid_from_string(mazes[i])
        add_agents(starting_pos, 1)
        env = Environment(
            grid,
            starting_pos,
            enable_observation=True,
            max_steps=1000,
            render_mode='human',
            observer=observer,
        )

        obs, infos = env.reset()
        print(obs['player_A'].flatten(), answers[i])
        assert all(np.array(obs['player_A'].flatten()) == np.array(answers[i]))

