from copy import deepcopy

import numpy as np

MINOTAUR = "minotaur"
EXIT = "exit"


class Environment:
    def __init__(self, transition_matrix=None, starting_pos=None, max_steps=100, render_mode=None):

        # Game Information
        self.transition_matrix = transition_matrix
        self.shape = transition_matrix.shape

        self.moves = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])

        # starting_pos = {
        #     minotaur: [x, y],
        #     EXIT: [x', y'],
        #     agent: [x", y"],
        # }
        self.starting_pos = starting_pos
        self.pos = None

        self.possible_agents = [
            agent for agent in starting_pos.keys() if agent != MINOTAUR and agent != EXIT
        ]
        self.agents = None

        # Simulation Information
        self.max_steps = max_steps
        self.timestep = 0

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

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
        infos[MINOTAUR] = self.pos[MINOTAUR].copy()

        if self.render_mode == "human":
            # TODO: self.draw_maze(), i.e. put the setting
            self.render()

        return observations, infos

    def _move_minotaur(self, steps=2):
        distances = {
            agent: abs(self.pos[MINOTAUR][0] - self.pos[agent][0]) + abs(self.pos[MINOTAUR][1] - self.pos[agent][1])
            for agent in self.agents
        }  # find manhattan distances between minotaur and agents

        if distances:
            min_agent = min(distances, key=distances.get)  # find the closest agent

            for i in range(steps):  # Move minotaur towards agent (2 steps for 1 agent step)
                transitions = self.transition_matrix[self.pos[MINOTAUR][0], self.pos[MINOTAUR][1], :]

                # First, horizontally
                if self.pos[MINOTAUR][1] < self.pos[min_agent][1] and transitions[1] == 1:
                    self.pos[MINOTAUR][1] += 1
                elif self.pos[MINOTAUR][1] > self.pos[min_agent][1] and transitions[3] == 1:
                    self.pos[MINOTAUR][1] -= 1
                # Then, vertically
                elif self.pos[MINOTAUR][0] < self.pos[min_agent][0] and transitions[2] == 1:
                    self.pos[MINOTAUR][0] += 1
                elif self.pos[MINOTAUR][0] > self.pos[min_agent][0] and transitions[0] == 1:
                    self.pos[MINOTAUR][0] -= 1

    def _agent_on_goal(self, agent):
        agent_position = self.pos[agent]
        shape = self.transition_matrix.shape

        return not (0 <= agent_position[0] <= shape[0] and 0 <= agent_position[1] <= shape[1])

    def _observe(self):
        # For each agent:
        # - Walls around it
        # - Walls around minotaur
        # - Distance to minotaur (manhattan distance)
        # - Distance to goal (manhattan distance)
        # - Direction of minotaur (north, east, south, west + sub-directions) -> [-1, 1]
        # - Direction of goal (north, east, south, west + sub-directions)

        walls_minotaur = np.array([*self.transition_matrix[*self.pos[MINOTAUR], :]]) == 0  # can't go -> walls

        observations = {}
        for agent in self.agents:
            walls_agent = np.array([*self.transition_matrix[*self.pos[agent], :]]) == 0

            distance_minotaur = abs(self.pos[MINOTAUR][0] - self.pos[agent][0]) + abs(
                self.pos[MINOTAUR][1] - self.pos[agent][1])
            distance_exit = abs(self.pos[EXIT][0] - self.pos[agent][0]) + abs(self.pos[EXIT][1] - self.pos[agent][1])

            # Return direction of minotaur and exit
            direction_minotaur = np.sign(self.pos[MINOTAUR] - self.pos[agent])
            direction_exit = np.sign(self.pos[EXIT] - self.pos[agent])

            observations[agent] = {
                "walls_agent": walls_agent,
                "walls_minotaur": walls_minotaur,
                "distance_minotaur": distance_minotaur,
                "distance_exit": distance_exit,
                "direction_minotaur": direction_minotaur,
                "direction_exit": direction_exit,
            }

        return observations

    def step(self, actions):
        """
        - 0 : delay
        - 1 : up
        - 2 : right
        - 3 : down
        - 4 : left

        MOVES: list = [[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]]

        :param actions: dictionary of actions to take for each agent
        :return:
        """

        # Move Agents
        for agent in self.agents:
            agent_action = actions[agent]
            agent_position = self.pos[agent]

            if agent_action and self.transition_matrix[*agent_position, agent_action - 1] == 1:
                self.pos[agent] += self.moves[agent_action]

        # Move minotaur
        self._move_minotaur(steps=2)

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}

        # Exit Maze Reward
        for agent in self.agents:
            if self._agent_on_goal(agent):
                rewards[agent] = 1
                terminations[agent] = True
                print(f"Player {agent} escaped!")

        # Check minotaur for Punishment
        for agent in self.agents:
            if all(self.pos[agent] == self.pos[MINOTAUR]):
                rewards[agent] = -1
                terminations[agent] = True

        # Not Terminated Punishment
        for agent in self.agents:
            if not terminations[agent]:
                rewards[agent] = -0.02

        self.timestep += 1

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep >= self.max_steps:
            rewards = {a: -0.02 for a in self.agents}
            truncations = {a: True for a in self.agents}

        if any(truncations.values()):
            self.agents = []

        # Remove agent from environment if terminated (avoid unnecessary computation)
        for agent in self.agents:
            if terminations[agent]:
                self.agents.remove(agent)

        observations = self._observe()

        # Get dummy infos (not used in this example)
        infos = {agent: self.pos[agent] for agent in self.agents}
        infos[MINOTAUR] = self.pos[MINOTAUR]

        # Rendering
        if self.render_mode == "human":
            # TODO: self.draw_players(), i.e. redraw the players
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == "human":
            # TODO: render current state
            # put the state history as variable in env to allow for animation of all states
            # raise NotImplementedError("Human rendering not implemented yet")
            return self._render_gui()
        elif self.render_mode == "ascii":
            return self._render_ascii()

    def _render_ascii(self):
        matrix = deepcopy(self.transition_matrix)

        # render_grid = np.zeros(tuple(matrix.shape[0:1]))
        item_grid = np.zeros(tuple(matrix.shape[0:2]))

        for item in self.pos:
            if item == MINOTAUR:
                item_grid[*self.pos[item]] = 100
            elif item == EXIT:
                item_grid[*self.pos[item]] = 10
            else:  # item == AGENT
                item_grid[*self.pos[item]] = 1

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
                        # elif item_grid[i, j] == 10:
                        #     buffer += "#"  # FIXME: causes the goal to be set at the wrong place with index -1
                        elif item_grid[i, j] == 100:
                            buffer += " M"
                        else:
                            buffer += "  "

                        if matrix[i, j, 1] == 0:
                            buffer += " █"
                        else:
                            buffer += " ░"

                    # buffer += " "

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

        dim_x, dim_y = self.transition_matrix.shape[0:2]

        # Init pygame
        if self.screen is None:
            pygame.init()

            self.screen = pygame.display.set_mode(((dim_x + 1) * self.CELL_SIZE, (dim_y + 1) * self.CELL_SIZE))
            pygame.display.set_caption("Minotaur's Maze")
            # self.clock = pygame.time.Clock()

        if self.screen is None:
            raise RuntimeError("Screen is None")

        # Handle events (this is really important, window won't show up otherwise)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Fill the background
        self.screen.fill(self.BG_COLOR)

        # Draw maze
        width, height = self.screen.get_size()

        offset_x = (width - dim_x * self.CELL_SIZE) // 2
        offset_y = (height - dim_y * self.CELL_SIZE) // 2

        surface = pygame.Surface((dim_x * self.CELL_SIZE + 1, dim_y * self.CELL_SIZE + 1))
        surface.fill((255, 255, 255))

        self._draw_maze(surface)

        # Draw players
        for agent in [agent for agent in self.pos if agent is not EXIT]:
            x, y = self.pos[agent]

            color = (0, 255, 0) if agent != MINOTAUR else (255, 0, 0)

            pygame.draw.circle(
                surface,
                color,
                (x * self.CELL_SIZE + self.CELL_SIZE / 2, y * self.CELL_SIZE + self.CELL_SIZE / 2),
                (self.CELL_SIZE - 10) // 2
            )

        self.screen.blit(surface, (offset_x, offset_y))

        # Update the display
        pygame.display.flip()

    def _draw_maze(self, surface):
        import pygame

        maze = self.transition_matrix

        def draw_line(start_pos, end_pos):
            pygame.draw.line(surface, self.WALL_COLOR, start_pos, end_pos, 1)

        for x, row in enumerate(maze):
            for y, cell in enumerate(row):
                for i, wall in enumerate(cell):
                    if wall == 0:
                        if i == 0:
                            draw_line(
                                (x * self.CELL_SIZE, y * self.CELL_SIZE),
                                (x * self.CELL_SIZE, (y + 1) * self.CELL_SIZE)
                            )
                        elif i == 1:
                            draw_line(
                                (x * self.CELL_SIZE, (y + 1) * self.CELL_SIZE),
                                ((x + 1) * self.CELL_SIZE, (y + 1) * self.CELL_SIZE)
                            )
                        elif i == 2:
                            draw_line(
                                ((x + 1) * self.CELL_SIZE, y * self.CELL_SIZE),
                                ((x + 1) * self.CELL_SIZE, (y + 1) * self.CELL_SIZE)
                            )
                        elif i == 3:
                            draw_line(
                                (x * self.CELL_SIZE, y * self.CELL_SIZE),
                                ((x + 1) * self.CELL_SIZE, y * self.CELL_SIZE)
                            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.quit()
