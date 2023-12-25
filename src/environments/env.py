from copy import deepcopy

import numpy as np

MINAUTOR = "minautor"
EXIT = "exit"


class Environment:
    def __init__(self, transition_matrix=None, starting_pos=None, max_steps=100):
        self.transition_matrix = transition_matrix

        self.moves = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]])

        # starting_pos = {
        #     MINAUTOR: [x, y],
        #     EXIT: [x', y'],
        #     agent: [x", y"],
        # }
        self.starting_pos = starting_pos
        self.pos = None

        self.possible_agents = [
            agent for agent in starting_pos.keys() if agent != MINAUTOR and agent != EXIT
        ]
        self.agents = None

        self.max_steps = max_steps
        self.timestep = 0

    def reset(self):
        self.agents = deepcopy(self.possible_agents)
        self.pos = deepcopy(self.starting_pos)

        self.timestep = 0

        observations = self._observe()

        infos = {agent: self.pos[agent] for agent in self.possible_agents}
        infos[MINAUTOR] = self.pos[MINAUTOR].copy()

        return observations, infos

    def _move_minautor(self, steps=2):
        distances = {
            agent: abs(self.pos[MINAUTOR][0] - self.pos[agent][0]) + abs(self.pos[MINAUTOR][1] - self.pos[agent][1])
            for agent in self.agents
        }  # find manhattan distances between minautor and agents

        if distances:
            min_agent = min(distances, key=distances.get)  # find the closest agent

            for i in range(steps):  # Move minautor towards agent (2 steps for 1 agent step)
                transitions = self.transition_matrix[self.pos[MINAUTOR][0], self.pos[MINAUTOR][1], :]

                # First, horizontally
                if self.pos[MINAUTOR][1] < self.pos[min_agent][1] and transitions[1] == 1:
                    self.pos[MINAUTOR][1] += 1
                elif self.pos[MINAUTOR][1] > self.pos[min_agent][1] and transitions[3] == 1:
                    self.pos[MINAUTOR][1] -= 1
                # Then, vertically
                elif self.pos[MINAUTOR][0] < self.pos[min_agent][0] and transitions[2] == 1:
                    self.pos[MINAUTOR][0] += 1
                elif self.pos[MINAUTOR][0] > self.pos[min_agent][0] and transitions[0] == 1:
                    self.pos[MINAUTOR][0] -= 1

    def _agent_on_goal(self, agent):
        agent_position = self.pos[agent]
        shape = self.transition_matrix.shape

        return not (0 <= agent_position[0] < shape[0] and 0 <= agent_position[1] < shape[1])

    def _observe(self):
        # For each agent:
        # - Walls around it
        # - Walls around Minautor
        # - Distance to minautor (manhattan distance)
        # - Distance to goal (manhattan distance)
        # - Direction of minautor (north, east, south, west + sub-directions) -> [-1, 1]
        # - Direction of goal (north, east, south, west + sub-directions)

        walls_minautor = np.array([*self.transition_matrix[*self.pos[MINAUTOR], :]]) == 0  # can't go -> walls

        observations = {}
        for agent in self.agents:
            walls_agent = np.array([*self.transition_matrix[*self.pos[agent], :]]) == 0

            distance_minautor = abs(self.pos[MINAUTOR][0] - self.pos[agent][0]) + abs(
                self.pos[MINAUTOR][1] - self.pos[agent][1])
            distance_exit = abs(self.pos[EXIT][0] - self.pos[agent][0]) + abs(self.pos[EXIT][1] - self.pos[agent][1])

            # Return direction of minautor and exit
            direction_minautor = np.sign(self.pos[MINAUTOR] - self.pos[agent])
            direction_exit = np.sign(self.pos[EXIT] - self.pos[agent])

            observations[agent] = {
                "walls_agent": walls_agent,
                "walls_minautor": walls_minautor,
                "distance_minautor": distance_minautor,
                "distance_exit": distance_exit,
                "direction_minautor": direction_minautor,
                "direction_exit": direction_exit,
            }

        return observations

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

            if agent_action != 4 and self.transition_matrix[*agent_position, agent_action] == 1:
                self.pos[agent] += self.moves[agent_action]

        # Move Minautor
        self._move_minautor(steps=2)

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}

        # Exit Maze Reward
        for agent in self.agents:
            if self._agent_on_goal(agent):
                rewards[agent] = 1
                terminations[agent] = True
                print(f"Player {agent} escaped!")

        # Check Minautor for Punishment
        for agent in self.agents:
            if all(self.pos[agent] == self.pos[MINAUTOR]):
                rewards[agent] = -1
                terminations[agent] = True

        # Not Terminated Punishment
        for agent in self.agents:
            if not terminations[agent]:
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

        # observations = self._observe()
        observations = {}

        # Get dummy infos (not used in this example)
        infos = {agent: self.pos[agent] for agent in self.possible_agents}
        infos[MINAUTOR] = self.pos[MINAUTOR]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        # TODO: render current state
        # put the state history as variable in env to allow for animation of all states

        matrix = deepcopy(self.transition_matrix)

        # render_grid = np.zeros(tuple(matrix.shape[0:1]))
        item_grid = np.zeros(tuple(matrix.shape[0:2]))

        for item in self.pos:
            if item == MINAUTOR:
                item_grid[*self.pos[item]] = 100
            # else:  # item == AGENT
            #     item_grid[*self.pos[item]] = 1
        for item in self.agents:
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
