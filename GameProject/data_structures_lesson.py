"""
Data Structures Behind Your IT - Interactive Education Agent
Extends the Game-like Animation example: the user now drives the lesson by
clicking buttons, and each data structure is explained with an IT analogy.
"""
import turtle
import random
import time
from collections import deque


class DataStructuresLesson:

    def setup_screen(self):
        """Set up the turtle graphics screen and the pen for drawing."""
        self.space = turtle.Screen()
        self.space.title("Data Structures Behind Your IT")
        self.space.bgcolor("lightblue")
        self.space.setup(width=1000, height=720)
        turtle.tracer(0)  # draw instantly; we call update() ourselves

        self.pen = turtle.Turtle()
        self.pen.hideturtle()
        self.pen.penup()

        self.buttons = []   # each button: (x, y, width, height, action)
        self.busy = False   # ignore clicks while the title is animating

        # Title, IT analogy and IT tip for each lesson
        self.lessons = {
            "Stack": ("STACK: Last In, First Out",
                      "IT analogy: your browser's Back button (and Undo) uses a stack.",
                      "IT Tip: Ctrl+Z pops your last action off the undo stack!"),
            "Queue": ("QUEUE: First In, First Out",
                      "IT analogy: a print queue, or a router's packet buffer.",
                      "IT Tip: a stuck print job blocks everything behind it - clear the queue!"),
            "Hash Table": ("HASH TABLE: Key -> Value",
                           "IT analogy: a DNS cache maps a domain name to its IP address.",
                           "IT Tip: 'ipconfig /flushdns' empties your computer's DNS cache!"),
            "Tree": ("TREE: Parent -> Children",
                     "IT analogy: your file system - folders inside folders.",
                     "IT Tip: a file path like C:/Users/alice is a route down the tree!"),
            "Graph": ("GRAPH: Nodes + Edges",
                      "IT analogy: the internet is a graph of routers (see Part 1(b))!",
                      "IT Tip: 'tracert google.com' shows the routers your packet passes!"),
        }

        # Demo data
        self.pages = ["google.com", "github.com", "python.org", "wikipedia.org", "youtube.com", "blackboard.edu"]
        self.dns = {"google.com": "142.250.80.46", "github.com": "140.82.112.3",
                    "python.org": "151.101.0.223", "amazon.com": "205.251.242.103",
                    "wikipedia.org": "208.80.154.224", "netflix.com": "54.237.226.164"}
        self.folders = {"C:/": ["Users", "Program Files", "Windows"],
                        "Users": ["alice", "bob"], "Program Files": ["Python", "Chrome"],
                        "Windows": ["System32"], "alice": ["Documents", "Downloads"],
                        "bob": ["Music"], "Python": ["python.exe"], "System32": ["drivers"]}
        self.routers = {"R1": (-300, 60), "R2": (-130, 160), "R3": (-130, -60),
                        "R4": (60, 100), "R5": (60, -120), "R6": (250, 20)}
        self.links = [("R1", "R2"), ("R1", "R3"), ("R2", "R4"), ("R3", "R4"),
                      ("R3", "R5"), ("R4", "R6"), ("R5", "R6")]

        self.space.onscreenclick(self.handle_click)

    # ------------------ Drawing helpers ------------------

    def write(self, x, y, text, color="black", size=14, style="normal", align="center"):
        self.pen.goto(x, y)
        self.pen.color(color)
        self.pen.write(text, align=align, font=("Arial", size, style))

    def draw_box(self, x, y, w, h, fill, label="", text_size=12):
        # Rectangle with bottom-left corner at (x, y) and an optional centered label
        self.pen.goto(x, y)
        self.pen.color("black", fill)
        self.pen.pendown()
        self.pen.begin_fill()
        for side in [w, h, w, h]:
            self.pen.forward(side)
            self.pen.left(90)
        self.pen.end_fill()
        self.pen.penup()
        if label:
            self.write(x + w / 2, y + h / 2 - text_size * 0.8, label, "black", text_size)

    def draw_button(self, x, y, w, h, label, action, fill="gold"):
        self.draw_box(x, y, w, h, fill, label, 14)
        self.buttons.append((x, y, w, h, action))

    def handle_click(self, cx, cy):
        # Check if the click landed inside any button's rectangle
        if self.busy:
            return
        for x, y, w, h, action in self.buttons:
            if x <= cx <= x + w and y <= cy <= y + h:
                action()
                return

    def typewriter(self, text, x, y):
        # Reveal the title one letter at a time (like the class example's animation)
        self.busy = True
        for i in range(1, len(text) + 1):
            self.pen.clear()
            self.write(x, y, text[:i], "darkblue", 26, "bold")
            self.space.update()
            time.sleep(0.05)
        self.busy = False

    # ------------------ Main menu ------------------

    def show_menu(self):
        self.pen.clear()
        self.buttons = []
        self.typewriter("Data Structures Behind Your IT", 0, 250)
        self.write(0, 200, "Click a topic to start a lesson", "black", 16, "italic")
        y = 110
        for name in self.lessons:
            self.draw_button(-120, y, 240, 50, name, lambda n=name: self.open_lesson(n), "lightyellow")
            y -= 70
        self.write(0, -300, "Every app you use is built on these 5 structures!", "darkgreen", 14, "bold")
        self.space.update()

    # ------------------ Lesson screens ------------------

    def open_lesson(self, name):
        # Reset the demo data, then draw the lesson with the typewriter title
        self.current = name
        self.message = "Click a button to try the demo!"
        self.stack = []
        self.queue = deque()
        self.job_number = 1
        self.buckets = [[] for i in range(5)]
        self.tree_depth = 1
        self.packet_path = []
        self.typewriter(self.lessons[name][0], 0, 280)
        self.draw_lesson()

    def draw_lesson(self):
        # Clear and redraw the whole lesson screen (simple redraw approach)
        title, analogy, tip = self.lessons[self.current]
        self.pen.clear()
        self.buttons = []
        self.write(0, 280, title, "darkblue", 26, "bold")
        self.write(0, 245, analogy, "black", 15, "italic")

        demos = {"Stack": self.draw_stack, "Queue": self.draw_queue,
                 "Hash Table": self.draw_hash_table, "Tree": self.draw_tree, "Graph": self.draw_graph}
        demos[self.current]()

        self.write(0, -250, self.message, "darkred", 14, "bold")
        self.write(0, -300, tip, "darkgreen", 14)
        self.draw_button(-460, 290, 100, 40, "Menu", self.show_menu, "lightgray")
        self.space.update()

    # ---- Stack: browser history ----
    def draw_stack(self):
        """Draw the browser history stack, with the top on the right."""
        self.draw_button(-420, 150, 160, 45, "Visit Page", self.stack_push)
        self.draw_button(-420, 85, 160, 45, "Back", self.stack_pop)
        y = -200
        for page in self.stack:  # bottom of stack drawn first
            self.draw_box(-100, y, 200, 45, "lightgreen", page)
            y += 50
        if self.stack:
            self.write(120, y - 40, "<- TOP", "red", 14, "bold", "left")
        else:
            self.write(0, -150, "(history is empty)", "gray", 14)

    def stack_push(self):
        """Simulate visiting a new page: push it onto the stack."""
        if len(self.stack) >= 7:
            self.message = "History is full in this demo - press Back!"
        else:
            page = random.choice(self.pages)
            self.stack.append(page)
            self.message = f"push('{page}'): new page goes on TOP"
        self.draw_lesson()

    def stack_pop(self):
        """Simulate pressing the Back button: pop the top page off the stack."""
        if not self.stack:
            self.message = "Nothing to go back to - the stack is empty!"
        else:
            page = self.stack.pop()
            self.message = f"pop() removed '{page}' from the TOP"
        self.draw_lesson()

    # ---- Queue: print jobs ----
    def draw_queue(self):
        """Draw the print queue, with the front on the left and the back on the right."""
        self.draw_button(-420, 150, 160, 45, "Add Job", self.queue_add)
        self.draw_button(-420, 85, 160, 45, "Print", self.queue_print)
        x = -330
        for job in self.queue:  # front of queue drawn first (left)
            self.draw_box(x, -60, 110, 60, "lightskyblue", job)
            x += 120
        if self.queue:
            self.write(-275, -100, "FRONT", "red", 14, "bold")
            self.write(x - 65, -100, "BACK", "blue", 14, "bold")
        else:
            self.write(0, -40, "(print queue is empty)", "gray", 14)

    def queue_add(self):
        """Add a new print job to the queue."""
        if len(self.queue) >= 6:
            self.message = "Queue is full in this demo - press Print!"
        else:
            job = f"Job {self.job_number}"
            self.job_number += 1
            self.queue.append(job)
            self.message = f"enqueue('{job}'): joins at the BACK"
        self.draw_lesson()

    def queue_print(self):
        if not self.queue:
            self.message = "Nothing to print - the queue is empty!"
        else:
            job = self.queue.popleft()
            self.message = f"dequeue() printed '{job}' from the FRONT"
        self.draw_lesson()

    # ---- Hash Table: DNS cache ----
    def draw_hash_table(self):
        self.draw_button(-420, 150, 160, 45, "Lookup", self.hash_lookup)
        y = 150
        for i, bucket in enumerate(self.buckets):
            self.draw_box(-200, y, 80, 50, "lightyellow", f"[{i}]")
            x = -110
            for domain in bucket:
                self.draw_box(x, y, 210, 50, "lightgreen", f"{domain} -> {self.dns[domain]}", 10)
                x += 220
            y -= 65

    def hash_lookup(self):
        """Simulate a DNS lookup: hash the domain name to find its bucket, then check if it's cached."""
        domain = random.choice(list(self.dns))
        hash_value = sum(ord(ch) for ch in domain)  # simple hash: add up the letter codes
        index = hash_value % len(self.buckets)
        if domain in self.buckets[index]:
            self.message = f"'{domain}': hash {hash_value} % 5 = {index} -> CACHE HIT, found instantly!"
        else:
            self.buckets[index].append(domain)
            self.message = f"'{domain}': hash {hash_value} % 5 = {index} -> cache miss, stored in bucket {index}"
        self.draw_lesson()

    # ---- Tree: file system ----
    def draw_tree(self):
        self.draw_button(-420, 150, 160, 45, "Expand", self.tree_expand)
        self.draw_button(-420, 85, 160, 45, "Collapse", self.tree_collapse)
        # Build the visible levels, one list of folder names per level
        levels = [["C:/"]]
        for depth in range(self.tree_depth):
            next_level = []
            for folder in levels[-1]:
                next_level += self.folders.get(folder, [])
            levels.append(next_level)
        # Spread each level across the screen, then draw lines from parent to child
        positions = {}
        y = 170
        for level in levels:
            gap = 760 / (len(level) + 1)
            for i, folder in enumerate(level):
                positions[folder] = (-280 + gap * (i + 1), y)
            y -= 100
        for parent, (px, py) in positions.items():
            for child in self.folders.get(parent, []):
                if child in positions:
                    cx, cy = positions[child]
                    self.pen.goto(px, py - 15)
                    self.pen.pendown()
                    self.pen.goto(cx, cy + 15)
                    self.pen.penup()
        for folder, (x, y) in positions.items():
            self.draw_box(x - 50, y - 15, 100, 30, "khaki", folder, 10)

    def tree_expand(self):
        if self.tree_depth >= 3:
            self.message = "Fully expanded! Every file has exactly one path from C:/"
        else:
            self.tree_depth += 1
            self.message = f"Expanded to level {self.tree_depth}: children appear under their parent folder"
        self.draw_lesson()

    def tree_collapse(self):
        if self.tree_depth > 1:
            self.tree_depth -= 1
        self.message = f"Collapsed to level {self.tree_depth}"
        self.draw_lesson()

    # ---- Graph: network of routers ----
    def draw_graph(self):
        self.draw_button(-460, 150, 170, 45, "Send Packet", self.send_packet)
        # Edges first, highlighted if they are on the packet's path
        path_links = list(zip(self.packet_path, self.packet_path[1:]))
        for a, b in self.links:
            on_path = (a, b) in path_links or (b, a) in path_links
            self.pen.color("red" if on_path else "gray")
            self.pen.pensize(6 if on_path else 2)
            self.pen.goto(self.routers[a])
            self.pen.pendown()
            self.pen.goto(self.routers[b])
            self.pen.penup()
        self.pen.pensize(1)
        # Then router nodes on top
        for name, (x, y) in self.routers.items():
            fill = "orange" if name in self.packet_path else "lightgreen"
            self.draw_box(x - 30, y - 20, 60, 40, fill, name, 14)

    def send_packet(self):
        source, target = random.sample(list(self.routers), 2)
        # Breadth-first search through the routers to find the fewest hops
        parents = {source: None}
        queue = deque([source])
        while queue:
            router = queue.popleft()
            neighbors = [b for a, b in self.links if a == router] + [a for a, b in self.links if b == router]
            for nxt in neighbors:
                if nxt not in parents:
                    parents[nxt] = router
                    queue.append(nxt)
        path = [target]
        while parents[path[-1]] is not None:
            path.append(parents[path[-1]])
        path.reverse()
        self.packet_path = path
        self.message = f"Packet {source} -> {target}: " + " > ".join(path) + f"  ({len(path) - 1} hops)"
        self.draw_lesson()


# ------------------ Driver Code ------------------
# Interactive: the user clicks to drive the lesson
lesson = DataStructuresLesson()
lesson.setup_screen()
lesson.show_menu()
turtle.done()
