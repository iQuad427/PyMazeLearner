simple_maze = """
+_+_+#+
|A+.+.|
+_+ + +
|M|.|.|
+ +_+ +
|.+.+.|
+_+_+_
"""

maze_1 = """
+_+_+_+_+#+_+
|A+.+.+.+.|.|
+_+ + + + + +
|.+.|.+.|.|.|
+ +_+ + +_+ +
|.|.|.|.+M+.|
+ + +_+ +_+ +
|.|.+.+.|.+.|
+ + +_+ + + +
|.|.|.+.+.+.|
+ + + + + + +
|.+.+.+.+.+.|
+_+_+_+_+_+_+
"""

maze_2 = """
+_+_+_+#+_+
|.|A+.+.+.|
+ + + +_+ +
|.+.|.+.|M|
+ +_+ + + +
|.+.+.|.+.|
+ +_+_+ + +
|.+.+.|.|.|
+ + + +_+ +
|.+.+.+.+.|
+_+_+_+_+_+
"""

maze_3 = """
+_+_+_+_+_+#+
|.+.+.+.|.+.|
+ + +_+ + + +
|.+.+.|.+.+.|
+ + + + + + +
|.+.+.+.+.|M|
+ + + + + + +
|.+.+.|.|.|.|
+ + + + + + +
|.|.+.+A+.+.|
+_+ + + + + +
|.+.|.|.+.+.|
+_+_+_+_+_+_+
"""

maze_4 = """
+_+_+_+_+_+_+_+_+
|.+A+.+.+.+.+.+.|
+ +_+_+ + + +_+ +
|.|.|.+.+.+.+.+.|
+_+ + +_+ +_+ + +
|.+.+.|.+.|.+.+.|
+ + + +_+ + + + +
|.|.+.+.|.|.|.+.|
+ +_+ +_+ +_+_+ +
|.|.|.+.|.|.+.+.|
+ + +_+_+_+ +_+ +
|.+.+.|.+.+.+.|.|
+ + +_+ + +_+ + +
|.+.|.+.|.|.|.+.|
+_+ +_+ +_+ +_+ +
|.+M+.|.+.+.+.+.|
+_+#+_+_+_+_+_+_+
"""

maze_5 = """
+_+_+_+_+_+_+_+_+
|.+.+.+.|.+.+.+.|
+ + + + + + +_+ +
|A+.+.|.+.+.|.+.|
+ + + + + +_+ + +
|.|.+.+.|.+.+.+.|
+ + + + + +_+ +_+
|.|.|.+.+.+.+.+.|
+_+ + +_+_+ +_+ +
|.+.|.+.+.+.+.+.|
+_+ + +_+ +_+_+ +
|.+.+.+.|.+.+.+.|
+ + +_+_+ + + + +
|.+.|.+.+.+.|.+.|
+ + +_+ + + + + +
|.+M+.|.+.+.+.+.|
+_+_+_+_+_+#+_+_+
"""

maze_6 = """
+_+_+_+_+_+_+_+_+
|.+.+.+.+.+.|.+.|
+_+ + + + + + + +
|.+.|.+.+.+.+.+.|
+ + + + +_+_+ + +
|.+.+.+.|.+.|.+.|
+ + +_+_+ + + +_+
|.+.|M|.|.+.+.+.|
+ +_+ + + + + +_+
|.+.+.+.+.|.|.+.|
+ + + + + + +_+ +
|.+.|.+.|.+A|.+.|
+ +_+_+_+ + +_+ +
|.+.|.+.+.|.+.|.|
+_+ + +_+_+ +_+ +
|.+.|.+.+.|.+.+.|
+_+_+_+_+#+_+_+_+
"""

maze_7 = """
+_+_+_+_+_+_+
|.+.+.+.+.+.|
+_+ + + + + +
|.|.+.+.|.|.|
+ + + + +_+ +
|.|.+.+.|.+.|
+ +_+ + +_+ +
#.|.+.+.+.+.|
+ + +_+ + + +
|.+.+A+.+.+.|
+ +_+_+ +_+ +
|.+.+M|.|.+.|
+_+_+_+_+_+_+
"""

maze_8 = """
+#+_+_+_+_+_+_+
|.|.+.+.+.+.+.|
+ +_+ +_+_+_+ +
|.|.|.+.+.+.+.|
+ + +_+_+_+_+ +
|.+.+.+.+.+.+.|
+ +_+ + + + + +
|.+.|.|M|.+.|.|
+ + + +_+ + + +
|.+.+.+A+.|.|.|
+ + +_+_+_+_+ +
|.+.+.+.+.|.+.|
+ +_+_+_+_+_+ +
|.+.+.+.+.+.+.|
+_+_+_+_+_+_+_+
"""

maze_9 = """
+_+#+_+_+_+_+
|.+.+.+.+.|.|
+ +_+ + +_+ +
|.+.+.|.+.+.|
+ + + + +_+ +
|.+.+.|.+.|.|
+ + + + + + +
|.|.+.+.|.|.|
+_+ + + +_+ +
|M+.+.+.+A|.|
+ + +_+ + + +
|.+.+.+.+.+.|
+_+_+_+_+_+_+
"""

maze_10 = """
+_+_+_+_+_+_+
|M+.+.+.+.+.|
+ +_+ +_+ + +
|.+.+.+.+.|.|
+ + + +_+ + +
|.|.|.+.+.|.#
+_+ + +_+ +_+
|.+.|.+.+.+.|
+ + + +_+ + +
|.|.|.+.+A+.|
+_+ +_+_+ + +
|.+.+.+.+.+.|
+_+_+_+_+_+_+
"""

maze_11 = """
+_+_+_+_+_+_+_+_+
#.|.+M+.+.+.+.+.|
+ + +_+ + + + +_+
|.+.+A+.+.|.+.+.|
+ + + + + + + + +
|.+.|.+.+.|.+.+.|
+ + + + + + +_+ +
|.+.|.+.+.|.+.+.|
+ + + + + + + + +
|.|.+.+.+.|.+.+.|
+ + + + + + + +_+
|.|.+.+.+.+.+.+.|
+ + + + +_+ + + +
|.|.+.|.+.|.+.|.|
+ + +_+ + + + +_+
|.+.+.+.+.+.+.+.|
+_+_+_+_+_+_+_+_+
"""

maze_12 = """
+_+_+#+_+_+_+_+_+
|.+.+.+.+.+.|.+.|
+ + +_+ + + + +_+
|M|.+.|.+.|.|.+.|
+_+_+ + +_+_+_+ +
|.|.+.+.|.|.+.+.|
+ + + + + + + + +
|.+.|.|.+.+.+.+.|
+ + +_+ + + + + |
|.+.+.+.+.+.+.+.|
+ +_+_+ + + + + +
|.+.|.+.+.|.+.+.|
+ + + + + + + + +
|.+.+.+.|.+.+.|.|
+ + + + + + +_+_+
|A+.+.+.+.+.+.+.|
+_+_+_+_+_+_+_+_+
"""

maze_13 = """
+_+_+_+_+_+_+_+_+
#.+.+.+.+.+.+.+.|
+ + + + + + + +_+
|.+.+.+.+.|.+.+.|
+ + + +_+ + + + +
|.|.+.+.|.+.+.|.|
+ + + + + + + + +
|.|.+.+.+.+.|.|.|
+ +_+ + + +_+ + +
|.|.|.|.+.+.|.|.|
+ + + + + + + + +
|.+.+M|.+.|.+.+.|
+_+ + + + + +_+ +
|.+.|.+A|.+.|.+.|
+ +_+ + +_+ + +_+
|.+.+.+.+.+.+.+.|
+_+_+_+_+_+_+_+_+
"""

maze_14 = """
+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
|.+.+.+.+.|.+.+.+.|.+.+.+.+.|
+_+_+ + + + +_+ +_+_+ +_+_+ +
|.|.|.|.|.|.|.+.+.+.+.|.+.+.#
+ + + +_+ + +_+ +_+_+_+_+_+ +
|.+.+.+.+.+.+.+.+.+.+.+.+.+.|
+ + +_+ + + +_+ + + +_+_+ + +
|.|.|.+.|.|.+.+.|.|.+.+.+.|.|
+ + +_+ + + + + + +_+_+_+ + +
|A|.|.+.|.|.|.|.|.+.+.+.|.|M|
+ + + + + + + + + +_+_+ + + +
|.|.|.|.+.|.|.+.+.+.+.+.|.|.|
+ + + + + + + + + +_+_+ + + +
|.|.|.|.|.|.|.|.|.+.+.+.|.|.|
+ + + + + + + + + +_+_+ + + +
|.|.|.|.|.|.|.|.|.+.+.+.|.|.|
+ + + + + + + + + +_+_+ + + +
|.+.+.+.+.+.+.+.+.+.+.+.+.+.|
+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
"""

maze_15 = """
+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
|.+.|.+.+.+.+.+.+.+.|.+.+.+.|
+ + + +_+ +_+_+ + +_+ +_+_+ +
|.|.+.+.|.|.+.+.|.+.|.|.|.|.|
+ +_+_+ + + +_+ + +_+ + + + +
|.+.|.+.+.+.+.+.+.+.|.|.+.+.|
+ + + +_+_+ +_+ +_+ + + + + +
|.|.+.+.+.|.|M+.+.|.+.|.|.|.|
+ + + + + + +_+ + + + + + + +
|.|.|.|.|.|.+.|.|.|.|.|.+.+.|
+ + + + + + + + + + + + +_+_+
|.+.+.+.+.|.|.+.|.|.|.|.|A|.#
+ +_+_+_+ + + + + + + + + + +
|.+.+.+.|.+.+.|.+.+.|.+.+.|.|
+_+_+ + +_+_+ + +_+ +_+ +_+ +
|.|.|.|.|.+.+.|.+.+.+.+.+.|.|
+ + + + +_+_+ + + +_+_+_+ + +
|.+.+.+.+.+.+.+.|.+.+.+.+.+.|
+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
"""

mazes = [
    maze_1,
    maze_2,
    maze_3,
    maze_4,
    maze_5,
    maze_6,
    maze_7,
    maze_8,
    maze_9,
    maze_10,
    maze_11,
    maze_12,
    maze_13,
    maze_14,
    maze_15,
]
