"""
Packet Race: BFS vs DFS vs A*
Extends the Athletics turtle race: 3 search algorithms race through the same
data-center maze to deliver a packet from the router (start) to the server (goal).
"""
import turtle
import random
import time
import heapq
from collections import deque


class PacketRaceGame:

    def setup_screen(self):
        self.space = turtle.Screen()
        self.space.title("Packet Race: BFS vs DFS vs A*")
        self.space.bgcolor("lightblue")
        self.space.setup(width=1300, height=850)
        turtle.tracer(0)  # draw instantly; we call update() ourselves

        # Pens for drawing the maze and writing text
        self.pen = turtle.Turtle()
        self.pen.hideturtle()
        self.pen.penup()
        self.writer = turtle.Turtle()
        self.writer.hideturtle()
        self.writer.penup()

        # The 3 racers: name, explored color, final path color, shape, maze center x
        self.racers = [
            {"name": "BFS", "explore": "lightskyblue", "path_color": "blue", "shape": "turtle", "x": -430},
            {"name": "DFS", "explore": "thistle", "path_color": "purple", "shape": "arrow", "x": 0},
            {"name": "A*", "explore": "palegreen", "path_color": "green", "shape": "classic", "x": 430},
        ]
        for racer in self.racers:
            racer["turtle"] = turtle.Turtle()
            racer["turtle"].shape(racer["shape"])
            racer["turtle"].color(racer["path_color"])
            racer["turtle"].penup()
            racer["wins"] = 0
            racer["total_steps"] = 0

        # What each winner teaches us
        self.lessons = {
            "BFS": "BFS wins: exploring level by level always finds the shortest route!",
            "DFS": "DFS wins: diving deep can get lucky, but it is not always the best route!",
            "A*": "A* wins: heuristics let AI agents search smarter, not harder!",
        }

        self.maze_pixels = 360   # each maze is drawn in a 360 x 360 box
        self.congestion_delay = 3  # extra steps a racer waits in a congested cell
        self.step_delay = 0.03     # seconds between animation steps

    # ------------------ Search algorithms (generators) ------------------
    # Each one yields (cell, path). path stays None until the goal is reached.

    def neighbors(self, cell, size, walls):
        r, c = cell
        result = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # right, down, left, up
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in walls:
                result.append((nr, nc))
        return result

    def build_path(self, parents, goal):
        # Walk backwards from the goal using the parent links
        path = [goal]
        while parents[path[-1]] is not None:
            path.append(parents[path[-1]])
        path.reverse()
        return path

    def bfs(self, size, walls, start, goal):
        queue = deque([start])  # FIFO queue
        parents = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == goal:
                yield cell, self.build_path(parents, goal)
                return
            yield cell, None
            for nxt in self.neighbors(cell, size, walls):
                if nxt not in parents:
                    parents[nxt] = cell
                    queue.append(nxt)

    def dfs(self, size, walls, start, goal):
        stack = [start]  # LIFO stack
        parents = {start: None}
        visited = set()
        while stack:
            cell = stack.pop()
            if cell in visited:
                continue
            visited.add(cell)
            if cell == goal:
                yield cell, self.build_path(parents, goal)
                return
            yield cell, None
            for nxt in self.neighbors(cell, size, walls):
                if nxt not in visited:
                    parents[nxt] = cell
                    stack.append(nxt)

    def astar(self, size, walls, start, goal):
        """A* search algorithm."""
        def manhattan(cell):
            return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

        heap = [(manhattan(start), 0, start)]  # (f = g + h, g, cell)
        parents = {start: None}
        best_g = {start: 0}
        visited = set()
        while heap:
            f, g, cell = heapq.heappop(heap)
            if cell in visited:
                continue
            visited.add(cell)
            if cell == goal:
                yield cell, self.build_path(parents, goal)
                return
            yield cell, None
            for nxt in self.neighbors(cell, size, walls):
                new_g = g + 1
                if nxt not in best_g or new_g < best_g[nxt]:
                    best_g[nxt] = new_g
                    parents[nxt] = cell
                    heapq.heappush(heap, (new_g + manhattan(nxt), new_g, nxt))

    # ------------------ Maze generation ------------------

    def make_maze(self, size, round_num):
        random.seed(round_num)  # fixed seed: same maze every time for this round
        start = (0, 0)
        goal = (size - 1, size - 1)
        while True:
            walls = set()
            for r in range(size):
                for c in range(size):
                    if (r, c) not in (start, goal) and random.random() < 0.28:
                        walls.add((r, c))
            open_cells = [(r, c) for r in range(size) for c in range(size)
                          if (r, c) not in walls and (r, c) not in (start, goal)]
            servers_down = set(random.sample(open_cells, 4))
            congested = set(random.sample([cell for cell in open_cells if cell not in servers_down], 5))

            # Use BFS to make sure the packet can still reach the server
            blocked = walls | servers_down
            for cell, path in self.bfs(size, blocked, start, goal):
                if path:
                    return walls, servers_down, congested, start, goal
            # No path found -> loop and generate a new maze

    # ------------------ Drawing helpers ------------------

    def cell_corner(self, racer, cell):
        # Top-left pixel of a cell inside this racer's maze
        left = racer["x"] - self.maze_pixels / 2
        top = 260
        return left + cell[1] * self.cell_size, top - cell[0] * self.cell_size

    def draw_cell(self, racer, cell, color):
        x, y = self.cell_corner(racer, cell)
        self.pen.goto(x, y)
        self.pen.color("gray", color)
        self.pen.pendown()
        self.pen.begin_fill()
        for i in range(4):
            self.pen.forward(self.cell_size)
            self.pen.right(90)
        self.pen.end_fill()
        self.pen.penup()

    def move_racer(self, racer, cell):
        x, y = self.cell_corner(racer, cell)
        racer["turtle"].goto(x + self.cell_size / 2, y - self.cell_size / 2)

    def write(self, x, y, text, color="black", size=12, style="normal", align="center"):
        self.writer.goto(x, y)
        self.writer.color(color)
        self.writer.write(text, align=align, font=("Arial", size, style))

    def draw_round(self, round_num, size, walls, servers_down, congested, start, goal):
        self.pen.clear()
        self.writer.clear()
        self.write(0, 370, "PACKET RACE: BFS vs DFS vs A*", "darkblue", 22, "bold")
        self.write(0, 340, f"Round {round_num} of 4  |  Data center maze {size} x {size}", "black", 14)
        self.write(0, -355, "Black = wall   Dark red = server down   Orange = congested (+3 wait)   "
                            "Gold = router (start)   Red = server (goal)", "black", 11)
        wins_text = "   ".join(f"{r['name']}: {r['wins']} wins" for r in self.racers)
        self.write(0, -385, "Total so far -> " + wins_text, "darkgreen", 12, "bold")

        # Draw every cell of all 3 mazes (server-down cells look normal at first)
        for racer in self.racers:
            self.write(racer["x"], 275, racer["name"], racer["path_color"], 16, "bold")
            for r in range(size):
                for c in range(size):
                    cell = (r, c)
                    color = "white"
                    if cell in walls:
                        color = "black"
                    elif cell in congested:
                        color = "orange"
                    elif cell == start:
                        color = "gold"
                    elif cell == goal:
                        color = "red"
                    self.draw_cell(racer, cell, color)
            self.move_racer(racer, start)
        self.space.update()
        time.sleep(1)

        # Servers go down at the start of the round -> they become walls
        self.write(0, 310, "ALERT: servers going down!", "darkred", 13, "bold")
        for racer in self.racers:
            for cell in servers_down:
                self.draw_cell(racer, cell, "darkred")
        self.space.update()
        time.sleep(1)

    # ------------------ One round of the race ------------------

    def run_round(self, round_num, size):
        walls, servers_down, congested, start, goal = self.make_maze(size, round_num)
        self.cell_size = self.maze_pixels // size
        self.draw_round(round_num, size, walls, servers_down, congested, start, goal)

        # Give each racer a fresh search generator on the SAME maze
        blocked = walls | servers_down
        searches = {"BFS": self.bfs, "DFS": self.dfs, "A*": self.astar}
        for racer in self.racers:
            racer["search"] = searches[racer["name"]](size, blocked, start, goal)
            racer["nodes"] = 0
            racer["steps"] = 0
            racer["wait"] = 0
            racer["path"] = None
            racer["finished"] = False

        # Race begins: advance every racer one step per loop (like the class example)
        while not all(racer["finished"] for racer in self.racers):
            for racer in self.racers:
                if racer["finished"]:
                    continue
                racer["steps"] += 1
                if racer["wait"] > 0:  # stuck in a congested cell
                    racer["wait"] -= 1
                    continue
                cell, path = next(racer["search"])
                racer["nodes"] += 1
                self.move_racer(racer, cell)
                if cell in congested:
                    racer["wait"] = self.congestion_delay
                elif cell not in (start, goal):
                    self.draw_cell(racer, cell, racer["explore"])
                if path:  # reached the server: draw the final route
                    racer["path"] = path
                    racer["finished"] = True
                    for step in path[1:-1]:
                        self.draw_cell(racer, step, racer["path_color"])
            self.space.update()
            time.sleep(self.step_delay)

        # Round winner = fewest steps (tie -> fewest nodes explored)
        winner = min(self.racers, key=lambda r: (r["steps"], r["nodes"]))
        winner["wins"] += 1
        self.show_scoreboard(winner)

    def show_scoreboard(self, winner):
        # Scoreboard under each maze
        for racer in self.racers:
            racer["total_steps"] += racer["steps"]
            x = racer["x"]
            self.write(x, -150, f"Nodes explored: {racer['nodes']}", "black", 12)
            self.write(x, -175, f"Path length: {len(racer['path']) - 1}", "black", 12)
            self.write(x, -200, f"Steps to finish: {racer['steps']}", "black", 12)
            if racer is winner:
                self.write(x, -235, "ROUND WINNER!", "goldenrod", 16, "bold")
        self.write(0, -280, f"{winner['name']} delivered the packet first!", "darkblue", 16, "bold")
        self.space.update()
        time.sleep(4)

    # ------------------ The full 4-round race ------------------

    def start_race(self):
        maze_sizes = [12, 14, 16, 18]  # the "400m": 4 rounds, maze grows each round
        for round_num in range(1, 5):
            self.run_round(round_num, maze_sizes[round_num - 1])

        # Final winner = most wins (tie -> fewest total steps)
        champion = min(self.racers, key=lambda r: (-r["wins"], r["total_steps"]))
        self.show_final_banner(champion)
        turtle.done()

    def show_final_banner(self, champion):
        self.pen.clear()
        self.writer.clear()
        for racer in self.racers:
            racer["turtle"].hideturtle()

        # Rectangular gold banner (same idea as the class example)
        self.pen.goto(-350, 250)
        self.pen.color("gold")
        self.pen.pensize(4)
        self.pen.pendown()
        for i in range(2):
            self.pen.forward(700)
            self.pen.right(90)
            self.pen.forward(330)
            self.pen.right(90)
        self.pen.penup()
        self.pen.pensize(1)

        # WINNER ANNOUNCEMENT
        self.write(0, 200, " THE WINNER IS! ", "gold", 28, "bold")
        self.write(0, 150, champion["name"], "darkblue", 26, "bold")
        self.write(0, 110, "WHAT THIS TEACHES:", "darkgreen", 14, "bold")
        self.write(0, 80, self.lessons[champion["name"]], "black", 13)

        # Final results table
        y = 30
        for racer in self.racers:
            self.write(0, y, f"{racer['name']}:  {racer['wins']} wins   |   {racer['total_steps']} total steps",
                       racer["path_color"], 14, "bold")
            y -= 30
        self.write(0, -120, "Packet Race: Smarter search = faster networks!", "red", 14, "italic")
        self.space.update()


# ----------- Driving the class -----------------------------------
handle = PacketRaceGame()
handle.setup_screen()
handle.start_race()
