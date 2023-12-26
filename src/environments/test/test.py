import sys

import pygame

# Set up Pygame
pygame.init()

# Constants
CELL_SIZE = 40
WALL_COLOR = (0, 0, 255)
BG_COLOR = (200, 200, 200)


# Function to draw the maze
def draw_maze(screen, maze):
    for x, row in enumerate(maze):
        for y, cell in enumerate(row):
            for i, wall in enumerate(cell):
                if wall == 0:
                    if i == 0:
                        pygame.draw.line(screen, WALL_COLOR, (x * CELL_SIZE, y * CELL_SIZE),
                                         (x * CELL_SIZE, (y + 1) * CELL_SIZE), 1)
                    elif i == 1:
                        pygame.draw.line(screen, WALL_COLOR, (x * CELL_SIZE, (y + 1) * CELL_SIZE),
                                         ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), 1)
                    elif i == 2:
                        pygame.draw.line(screen, WALL_COLOR, ((x + 1) * CELL_SIZE, y * CELL_SIZE),
                                         ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), 1)
                    elif i == 3:
                        pygame.draw.line(screen, WALL_COLOR, (x * CELL_SIZE, y * CELL_SIZE),
                                         ((x + 1) * CELL_SIZE, y * CELL_SIZE), 1)


# Sample 3D array representing the maze
maze_data = [
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

# Create the screen
screen = pygame.display.set_mode(((len(maze_data) + 1) * CELL_SIZE, (len(maze_data[0]) + 1) * CELL_SIZE))
pygame.display.set_caption("Maze Drawing")

# Compute offset to put the maze in the center of the screen
WIDTH, HEIGHT = screen.get_size()
offset_x = (WIDTH - len(maze_data) * CELL_SIZE) // 2
offset_y = (HEIGHT - len(maze_data[0]) * CELL_SIZE) // 2

# Main loop
i = 0
running = True
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background
    screen.fill(BG_COLOR)

    # Draw the maze on a surface
    surface = pygame.Surface((len(maze_data) * CELL_SIZE + 1, len(maze_data[0]) * CELL_SIZE + 1))
    surface.fill((255, 255, 255))
    draw_maze(surface, maze_data)

    # Blit the surface onto the screen
    screen.blit(surface, (offset_x, offset_y))

    # Add an agent at (x, y) = (0, 0)
    x, y = (0, 0) if i < 20 else (1, 0)
    print(x, y)
    pygame.draw.circle(screen, (255, 0, 0), (offset_x + x * CELL_SIZE + CELL_SIZE/2, offset_y + y * CELL_SIZE + CELL_SIZE/2), (CELL_SIZE-10) // 2)

    # Update the display
    pygame.display.flip()
    # clock.tick(30)

    # Update the rendering
    pygame.time.wait(100)

    i += 1

# Quit Pygame
pygame.quit()
sys.exit()
