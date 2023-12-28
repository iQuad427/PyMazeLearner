from copy import deepcopy
from os import path
from typing import Dict, Union

import numpy as np

from src.environments.models import Objects, BaseView
from src.environments.observer.default import DefaultObserver
from src.environments.utils import manhattan_distance, _star


class Environment:
    def __init__(
        self,
        transition_matrix=None,
        starting_pos=None,
        max_steps=100,
        render_mode=None,
        enable_observation=True,
        observer=None,
    ):
        if observer is None:
            observer = DefaultObserver()
        self.observer = observer

        # Game Information
        self.minotaur_img = None
        self.enable_observation = enable_observation
        self.transition_matrix = transition_matrix
        self.shape = transition_matrix.shape

        self.moves = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]])

        # starting_pos = {
        #     minotaur: [x, y],
        #     Objects.EXIT: [x', y'],
        #     agent: [x", y"],
        # }
        self.starting_pos = starting_pos
        self.pos = None

        # Exclude all objects that are not agents. (Minotaur is controlled by the environment and exit is static)
        self.possible_agents = [
            agent
            for agent in starting_pos.keys()
            if agent != Objects.EXIT and agent != Objects.MINOTAUR and agent != Objects.AGENT
        ]
        self.agents = None

        # Simulation Information
        self.max_steps = max_steps
        self.timestep = 0

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.surface = None
        self.clock = None

        # Sprites
        self.ice_img = None
        self.player_img = None

        self.CELL_SIZE = 40
        self.WALL_COLOR = (0, 0, 255)
        self.BG_COLOR = (200, 200, 200)

        self.render_fps = 4

    def reset(self):
        self.agents = deepcopy(self.possible_agents)
        self.pos = deepcopy(self.starting_pos)

        self.timestep = 0

        observations = self._observe()

        infos = {agent: self.pos[agent] for agent in self.possible_agents}
        infos[Objects.MINOTAUR] = self.pos[Objects.MINOTAUR].copy()

        if self.render_mode == "human":
            self.render()

        return observations, infos

    def _move_minotaur(self, steps=2):
        distances = {
            agent: manhattan_distance(self.pos[Objects.MINOTAUR], self.pos[agent])
            for agent in self.agents
        }  # Find manhattan distances between minotaur and agents

        if distances:
            min_agent = min(distances, key=distances.get)  # find the closest agent

            for i in range(
                steps
            ):  # Move minotaur towards agent (2 steps for 1 agent step)
                transitions = self.transition_matrix[
                    self.pos[Objects.MINOTAUR][0], self.pos[Objects.MINOTAUR][1], :
                ]

                # First, horizontally
                if (
                    self.pos[Objects.MINOTAUR][1] < self.pos[min_agent][1]
                    and transitions[1] == 1
                ):
                    self.pos[Objects.MINOTAUR][1] += 1
                elif (
                    self.pos[Objects.MINOTAUR][1] > self.pos[min_agent][1]
                    and transitions[3] == 1
                ):
                    self.pos[Objects.MINOTAUR][1] -= 1
                # Then, vertically
                elif (
                    self.pos[Objects.MINOTAUR][0] < self.pos[min_agent][0]
                    and transitions[2] == 1
                ):
                    self.pos[Objects.MINOTAUR][0] += 1
                elif (
                    self.pos[Objects.MINOTAUR][0] > self.pos[min_agent][0]
                    and transitions[0] == 1
                ):
                    self.pos[Objects.MINOTAUR][0] -= 1

    def _agent_on_goal(self, agent) -> bool:
        """
        Check if the agent is on the goal position (i.e. the exit)
        :param agent: agent to check
        :return: True if the agent is on the goal position, False otherwise
        """

        agent_position = self.pos[agent]
        shape = self.transition_matrix.shape

        return not (
            0 <= agent_position[0] < shape[0] and 0 <= agent_position[1] < shape[1]
        )

    def _observe(self) -> Union[Dict[str, BaseView], None]:
        """
        For each agent, return a BaseView object containing the following information:
        - walls_agent: 4x1 array of booleans, indicating whether there is a wall in the direction of the agent
        - walls_minotaur: 4x1 array of booleans, indicating whether there is a wall in the direction of the minotaur
        - distance_minotaur: Manhattan distance between the agent and the minotaur
        - distance_exit: Manhattan distance between the agent and the exit
        - direction_minotaur: 2x1 array of booleans, indicating the direction of the minotaur
        - direction_exit: 2x1 array of booleans, indicating the direction of the exit

        :return: dictionary of BaseView objects
        """

        if not self.enable_observation:
            return None

        return self.observer.get_observation(self)

    def step(self, actions):
        """
        - 0 : up
        - 1 : right
        - 2 : down
        - 3 : left
        - 4 : delay

        MOVES: list = [[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]]

        :param actions: dictionary of actions to take for each agent
        :return:
        """

        # Move Agents
        for agent in self.agents:
            agent_action = actions[agent]
            agent_position = self.pos[agent]

            if (
                agent_action != 4
                and _star(agent_position, self.transition_matrix)[agent_action]
                == Objects.EMPTY
            ):
                self.pos[agent] += self.moves[agent_action]

        # Move minotaur
        self._move_minotaur(steps=2)

        # Check termination conditions
        terminations = {a: False for a in self.possible_agents}
        rewards = {a: 0 for a in self.possible_agents}

        # Exit Maze Reward
        for agent in self.agents:
            # Check agent for Reward
            if self._agent_on_goal(agent):
                rewards[agent] = 1
                terminations[agent] = True
                if self.render_mode == "human":
                    print(f"Player {agent} escaped!")
            # Check minotaur for Punishment
            elif np.array_equal(self.pos[agent], self.pos[Objects.MINOTAUR]):
                rewards[agent] = -1
                terminations[agent] = True
                if self.render_mode == "human":
                    print(f"Player {agent} got eaten!")
            else:
                rewards[agent] = -0.02

        # Remove agent from environment if terminated (avoid unnecessary computation)
        for agent in self.agents:
            if terminations[agent]:
                self.agents.remove(agent)

        self.timestep += 1

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.possible_agents}

        if self.timestep >= self.max_steps:
            rewards = {a: -0.02 for a in self.possible_agents}
            truncations = {a: True for a in self.possible_agents}

        if any(truncations.values()):
            self.agents = []

        observations = self._observe()

        infos = {agent: self.pos[agent] for agent in self.possible_agents}
        infos[Objects.MINOTAUR] = self.pos[Objects.MINOTAUR]

        # Rendering
        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == "human":
            return self._render_gui()
        elif self.render_mode == "ascii":
            return self._render_ascii()

    def _render_ascii(self):
        matrix = deepcopy(self.transition_matrix)

        item_grid = np.zeros(tuple(matrix.shape[0:2]))

        for item in self.pos:
            if item == Objects.MINOTAUR:
                item_grid[self.pos[item][0], self.pos[item][1]] = 100
        for item in self.agents:
            item_grid[self.pos[item][0], self.pos[item][1]] = 1

        buffer = ""

        for i, row in enumerate(self.transition_matrix):
            for wall_iter in range(2):
                for j, col in enumerate(row):
                    if j == 0:
                        if col[3] == 0:
                            buffer += "█"
                        else:
                            buffer += "#"

                    if wall_iter == 0:
                        if matrix[i, j, 0] == 0:
                            buffer += " █"
                        else:
                            if i == 0:
                                buffer += " G"
                            else:
                                buffer += " ░"

                        buffer += " █"
                    else:
                        if item_grid[i, j] == 1:
                            buffer += " A"
                        elif item_grid[i, j] == 100:
                            buffer += " M"
                        else:
                            buffer += "  "

                        if matrix[i, j, 1] == 0:
                            buffer += " █"
                        else:
                            buffer += " ░"

                buffer += "\n"

            if i == len(row) - 1:
                buffer += "█"
                for j, col in enumerate(row):
                    if matrix[i, j, 2] == 0:
                        buffer += " █"
                    else:
                        buffer += " #"

                    buffer += " █"

                buffer += " "

        return buffer

    def _render_gui(self):
        import pygame  # import now to avoid making it mandatory to run code without human rendering

        dim_y, dim_x = self.transition_matrix.shape[0:2]

        # Init pygame
        if self.screen is None:
            pygame.init()

            self.screen = pygame.display.set_mode(
                ((dim_x + 1) * self.CELL_SIZE, (dim_y + 1) * self.CELL_SIZE)
            )
            pygame.display.set_caption("Minotaur's Maze")

            # Fill the background
            self.screen.fill(self.BG_COLOR)

        if self.screen is None:
            raise RuntimeError("Screen is None")

        # Handle events (this is really important, window won't show up otherwise)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        width, height = self.screen.get_size()

        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), (self.CELL_SIZE, self.CELL_SIZE)
            )

        if self.player_img is None:
            file_name = path.join(path.dirname(__file__), "img/cookie.png")
            self.player_img = pygame.transform.scale(
                pygame.image.load(file_name), (self.CELL_SIZE + 10, self.CELL_SIZE + 10)
            )

        if self.minotaur_img is None:
            file_name = path.join(path.dirname(__file__), "img/elf_down.png")
            self.minotaur_img = pygame.transform.scale(
                pygame.image.load(file_name), (self.CELL_SIZE + 10, self.CELL_SIZE + 10)
            )

        offset_x = (width - dim_x * self.CELL_SIZE) // 2
        offset_y = (height - dim_y * self.CELL_SIZE) // 2

        # Draw maze
        self.surface = pygame.Surface(
            (dim_x * self.CELL_SIZE + 1, dim_y * self.CELL_SIZE + 1)
        )
        self.surface.fill((255, 255, 255))

        self._draw_maze(self.surface)

        # Draw players
        for agent in [agent for agent in self.pos if agent is not Objects.EXIT and agent is not Objects.AGENT]:
            x, y = self.pos[agent]

            # Render player image
            self.surface.blit(
                self.player_img if agent != Objects.MINOTAUR else self.minotaur_img,
                (y * self.CELL_SIZE - 5, x * self.CELL_SIZE - 5),
            )

        self.screen.blit(self.surface, (offset_x, offset_y))

        # Update the display
        pygame.display.flip()

    def _draw_maze(self, surface):
        import pygame

        maze = self.transition_matrix

        def draw_line(start_pos, end_pos):
            pygame.draw.line(surface, self.WALL_COLOR, start_pos, end_pos, 1)

        for x, row in enumerate(maze):
            for y, cell in enumerate(row):
                surface.blit(self.ice_img, (y * self.CELL_SIZE, x * self.CELL_SIZE))
                for i, wall in enumerate(cell):
                    if wall == 0:
                        if i == 0:
                            draw_line(
                                (y * self.CELL_SIZE, x * self.CELL_SIZE),
                                ((y + 1) * self.CELL_SIZE, x * self.CELL_SIZE),
                            )
                        elif i == 1:
                            draw_line(
                                ((y + 1) * self.CELL_SIZE, x * self.CELL_SIZE),
                                ((y + 1) * self.CELL_SIZE, (x + 1) * self.CELL_SIZE),
                            )
                        elif i == 2:
                            draw_line(
                                ((y + 1) * self.CELL_SIZE, (x + 1) * self.CELL_SIZE),
                                (y * self.CELL_SIZE, (x + 1) * self.CELL_SIZE),
                            )
                        elif i == 3:
                            draw_line(
                                (y * self.CELL_SIZE, (x + 1) * self.CELL_SIZE),
                                (y * self.CELL_SIZE, x * self.CELL_SIZE),
                            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.quit()
